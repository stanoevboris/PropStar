import itertools
import queue
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from tqdm import tqdm

from utils import OrderedDictList, is_imbalanced, balance_dataset_with_smote
from vectorizers import conjunctVectorizer

from woe import WOEEncoder

import logging
import sqlalchemy as sa
from db_utils import MSSQLDatabase, MYSQLDatabase

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

SQL_TYPE_DICT = {
    "mssql": MSSQLDatabase,
    "mysql": MYSQLDatabase
}


def get_data(sql_type: str, database: Optional[str], target_schema: str, include_all_schemas: Optional[bool]):
    db_object = SQL_TYPE_DICT[sql_type]

    kwargs = {
        "target_schema": target_schema
    }

    if sql_type == "mssql":
        kwargs["database"] = database
        kwargs["include_all_schemas"] = include_all_schemas

    db = db_object(**kwargs)
    engine = sa.create_engine(db.connection_url, echo=True)
    tables_dict = {}
    with engine.connect() as connection:
        schemas = [target_schema] if not include_all_schemas else sa.inspect(engine).get_schema_names()

        for schema in schemas:
            tables = sa.inspect(engine).get_table_names(schema=schema)
            for table in tables:
                tables_dict[table] = db.get_table(schema=schema,
                                                  table_name=table,
                                                  connection=connection)

        logging.info(f"Total tables read: {len(tables_dict)}")
        logging.info(f"Tables read: {list(tables_dict.keys())}")

        pks = pd.read_sql(sa.text(db.get_primary_keys()), connection)
        fks = pd.read_sql(sa.text(db.get_foreign_keys()), connection)

    # Conversion for PKs and FKs in order to match the source code format of the variables
    pks_dict = dict(zip(pks['TableName'], pks['PrimaryKeyColumn']))
    fk_graph = fks[['ChildTable', 'ChildColumn', 'ReferencedTable', 'ReferencedColumn']].values.tolist()
    return tables_dict, pks_dict, fk_graph


def get_table_keys(quadruplet):
    """
    A basic method for gaining a given table's keys.
    """

    tk = defaultdict(set)
    for entry in quadruplet:
        tk[entry[0]].add(entry[1])
        tk[entry[2]].add(entry[3])
    return tk


def generate_relational_words(tables,
                              fkg,
                              target_table=None,
                              target_attribute=None,
                              relation_order=(2, 4),
                              indices=None,
                              encoder=None,
                              vectorization_type="tfidf",
                              num_features=10000,
                              primary_keys=None):
    """
    Key method for generation of relational words and documents.
    It traverses individual tables in path, and consequantially appends the witems to a witem set.
    This method is a rewritten,
        non-exponential (in space) version of the original Wordification algorithm (Perovsek et al, 2014).
    input: a collection of tables and a foreign key graph
    output: a representation in form of a sparse matrix.
    """

    fk_graph = nx.Graph(
    )  # a simple undirected graph as the underlying fk structure
    core_foreign_keys = set()
    all_foreign_keys = set()

    for foreign_key in fkg:

        # foreing key mapping
        t1, k1, t2, k2 = foreign_key

        if t1 == target_table:
            core_foreign_keys.add(k1)

        elif t2 == target_table:
            core_foreign_keys.add(k2)

        all_foreign_keys.add(k1)
        all_foreign_keys.add(k2)

        # add link, note that this is in fact a typed graph now
        fk_graph.add_edge((t1, k1), (t2, k2))

    # this is more efficient than just orderedDict object
    feature_vectors = OrderedDictList()
    if indices is not None:
        core_table = tables[target_table].iloc[indices, :]
    else:
        core_table = tables[target_table]
    all_table_keys = get_table_keys(fkg)
    core_foreign = None
    target_classes = core_table[target_attribute]

    # This is a remnant of one of the experiment, left here for historical reasons :)
    if target_attribute == "Delka_hospitalizace":
        tars = []
        for tc in target_classes:
            if int(tc) >= 10:
                tars.append(0)
            else:
                tars.append(1)
        target_classes = pd.DataFrame(np.array(tars))
        print(np.sum(tars) / len(target_classes))

    total_witems = set()
    num_witems = 0

    # The main propositionalization routine
    logging.info("Propositionalization of core table ..")
    for index, row in tqdm(core_table.iterrows(),
                           total=core_table.shape[0]):
        for i in range(len(row)):
            column_name = row.index[i]
            if column_name != target_attribute and not column_name in core_foreign_keys:
                witem = "-".join([target_table, column_name, str(row.iloc[i])])
                feature_vectors[index].append(witem)
                num_witems += 1
                total_witems.add(witem)

    logging.info("Traversing other tables ..")
    for core_fk in core_foreign_keys:  # this is normaly a single key.
        bfs_traversal = dict(
            nx.bfs_successors(fk_graph, (target_table, core_fk)))

        # Traverse the row space
        for index, row in tqdm(core_table.iterrows(),
                               total=core_table.shape[0]):

            current_depth = 0
            to_traverse = queue.Queue()
            to_traverse.put(target_table)  # seed table
            max_depth = 2
            tables_considered = 0
            parsed_tables = set()

            # Perform simple search
            while current_depth < max_depth:
                current_depth += 1
                origin = to_traverse.get()
                if current_depth == 1:
                    successor_tables = bfs_traversal[(origin, core_fk)]
                else:
                    if origin in bfs_traversal:
                        successor_tables = bfs_traversal[origin]
                    else:
                        continue
                for succ in successor_tables:
                    to_traverse.put(succ)
                for table in successor_tables:
                    if (table) in parsed_tables:
                        continue
                    parsed_tables.add(table)
                    first_table_name, first_table_key = origin, core_fk
                    next_table_name, next_table_key = table
                    if first_table_name not in tables or next_table_name not in tables:
                        continue

                    # link and generate witems
                    first_table = tables[first_table_name]
                    second_table = tables[next_table_name]
                    if first_table_name == target_table:
                        key_to_compare = row[first_table_key]
                    elif first_table_name != target_table and current_depth == 2:
                        key_to_compare = None
                        for edge in fk_graph.edges():
                            if edge[0][0] == target_table and edge[1][0] == first_table_name:
                                key_to_compare = first_table[first_table[
                                                                 edge[1][1]] == row[edge[0][1]]][first_table_key]
                        if not key_to_compare is None:
                            pass
                        else:
                            continue

                    # The second case
                    trow = second_table[second_table[next_table_key] == key_to_compare]
                    for x in trow.columns:
                        if not x in all_foreign_keys and x != target_attribute:
                            for value in trow[x]:
                                witem = "-".join(
                                    str(x)
                                    for x in [next_table_name, x, value])
                                total_witems.add(witem)
                                num_witems += 1
                                feature_vectors[index].append(witem)

    # Summary of the output
    logging.info("Stored {} witems..".format(num_witems))
    logging.info("Learning representation from {} unique witems.".format(
        len(total_witems)))

    # Encoder is an arbitrary vectorizer, some of the well known ones are implemented here, it's simple to add
    # your own!
    if encoder:
        matrix = relational_words_to_matrix_with_vec(
            feature_vectors, encoder, vectorization_type=vectorization_type)
        return matrix, target_classes.array
    else:
        matrix, encoder = relational_words_to_matrix(
            feature_vectors,
            relation_order,
            vectorization_type,
            max_features=num_features)
        # train_data imbalanced labels handling
        if is_imbalanced(target_classes, threshold=0.3):
            features, labels = balance_dataset_with_smote(matrix, target_classes.array)
        else:
            features, labels = matrix, target_classes.array

        logging.info("Stored sparse representation of the witemsets.")

        return features, labels, encoder


def relational_words_to_matrix_with_vec(fw,
                                        encoder,
                                        vectorization_type="tfidf"):
    """
    Just do the transformation. This is for proper cross-validation (on the test set)
    """

    docs = []
    if vectorization_type == "tfidf" or vectorization_type == "binary":
        for k, v in fw.items():
            docs.append(set(v))
        mtx = encoder.transform(docs)
    else:
        for k, v in fw.items():
            docs.append(" ".join(v))
        mtx = encoder.transform(docs)

    return mtx


def relational_words_to_matrix(fw,
                               relation_order,
                               vectorization_type="tfidf",
                               max_features=10000):
    """
    Employ the conjuncVectorizer to obtain zero order features.
    input: documents
    output: a sparse matrix
    """

    docs = []

    if vectorization_type == "tfidf" or vectorization_type == "binary":
        if vectorization_type == "tfidf":
            vectorizer = conjunctVectorizer(max_atoms=relation_order,
                                            max_features=max_features)
        elif vectorization_type == "binary":
            vectorizer = conjunctVectorizer(max_atoms=relation_order,
                                            binary=True,
                                            max_features=max_features)
        for k, v in fw.items():
            docs.append(set(v))
        mtx = vectorizer.fit_transform(docs)

    elif vectorization_type == "sklearn_tfidf" \
            or vectorization_type == "sklearn_binary" \
            or vectorization_type == "sklearn_hash":

        if vectorization_type == "sklearn_tfidf":
            vectorizer = TfidfVectorizer(max_features=max_features,
                                         binary=True)
        elif vectorization_type == "sklearn_binary":
            vectorizer = TfidfVectorizer(max_features=max_features,
                                         binary=False)
        elif vectorization_type == "sklearn_hash":
            vectorizer = HashingVectorizer()

        for k, v in fw.items():
            docs.append(" ".join(v))

        mtx = vectorizer.fit_transform(docs)
    return mtx, vectorizer


def generate_custom_relational_words(tables,
                                     fkg,
                                     target_table=None,
                                     target_attribute=None,
                                     relation_order=(2, 4),
                                     indices=None,
                                     encoder=None,
                                     vectorization_type="tfidf",
                                     num_features=10000,
                                     primary_keys=None):
    """
    Key method for generation of relational words and documents.
    It traverses individual tables in path, and consequentially appends the witems to a witem set.
    input: a collection of tables and a foreign key graph
    output: a representation in form of a sparse matrix.
    """

    fk_graph = nx.Graph()  # a simple undirected graph as the underlying fk structure
    core_foreign_keys = set()
    all_foreign_keys = set()
    joined_tables = list()
    for foreign_key in fkg:

        # foreign key mapping
        t1, k1, t2, k2 = foreign_key
        if t1 == 'SalesOrderHeaderSalesReason':
            continue

        if t1 == target_table:
            core_foreign_keys.add(k1)

        elif t2 == target_table:
            core_foreign_keys.add(k2)

        all_foreign_keys.add(k1)
        all_foreign_keys.add(k2)

        # add link, note that this is in fact a typed graph now
        fk_graph.add_edge((t1, k1), (t2, k2))

    if not indices is None:
        core_table = tables[target_table].iloc[indices]
    else:
        core_table = tables[target_table]

    features_data = core_table.copy()

    logging.info("Starting to join tables ..")
    for core_fk in core_foreign_keys:  # this is normally a single key.
        bfs_traversal = dict(
            nx.bfs_successors(fk_graph, (target_table, core_fk)))

        # Traverse the row space
        current_depth = 0
        to_traverse = queue.Queue()
        to_traverse.put(target_table)  # seed table
        max_depth = 2
        tables_considered = 0
        parsed_tables = set()

        while current_depth < max_depth:
            current_depth += 1
            origin = to_traverse.get()
            if current_depth == 1:
                successor_tables = bfs_traversal[(origin, core_fk)]
            else:
                if origin in bfs_traversal:
                    successor_tables = bfs_traversal[origin]
                else:
                    continue
            for succ in successor_tables:
                to_traverse.put(succ)
            for table in successor_tables:
                if table in parsed_tables:
                    continue
                parsed_tables.add(table)
                first_table_name, first_table_key = origin, core_fk
                next_table_name, next_table_key = table
                if not first_table_name in tables or not next_table_name in tables:
                    continue

                # link and generate witems
                second_table = tables[next_table_name]
                joined_tables.append(next_table_name)
                features_data = features_data.merge(second_table,
                                                    how='inner',
                                                    left_on=first_table_key,
                                                    right_on=next_table_key,
                                                    suffixes=(None, f'__y'))

                columns_to_drop = {f"{key}__y" for key in itertools.chain(all_foreign_keys, primary_keys.values()) if
                                   f"{key}__y" not in core_foreign_keys and f"{key}__y" in features_data.columns}
                columns_to_drop_special_case = {key for key in all_foreign_keys if
                                                key not in core_foreign_keys and key in features_data.columns}
                columns_to_drop |= columns_to_drop_special_case
                features_data.drop(list(columns_to_drop), axis=1, inplace=True)

    # drop all keys
    available_keys = {key for key in all_foreign_keys
                      if key in features_data.columns and key != primary_keys[target_table]}
    available_keys |= {value for key, value in primary_keys.items()
                       if key != target_table
                       and value != primary_keys[target_table]
                       and value in features_data.columns}
    features_data.drop(list(available_keys), axis=1, inplace=True)
    # features_data.drop(target_attribute, axis=1, inplace=True)
    features_data = features_data.loc[:, ~features_data.columns.str.endswith('__y')]
    # features_data = features_data.apply(pd.to_numeric, errors='ignore')
    try:
        # features_data.set_index(primary_keys[target_table], inplace=True)
        features_data.reset_index(drop=True, inplace=True)
    except Exception as es:
        print(es)

    try:
        target_classes_dict = dict(zip(features_data.index,
                                       features_data[target_attribute]))
    except Exception as es:
        print(es)
        target_classes_dict = dict(zip(features_data[primary_keys[target_table]], features_data[target_attribute]))

    target_classes_array = [target_classes_dict[index] for index in features_data.index]
    features_data.drop(target_attribute, axis=1, inplace=True)
    features_data.drop(primary_keys[target_table], axis=1, inplace=True)
    if encoder:
        transformed_features = calculate_features(features_data=features_data,
                                                  encoder=encoder)

        logging.info("Stored woe representation of the features.")

        return transformed_features, target_classes_dict
    else:
        transformed_features, encoder = calculate_features(features_data=features_data,
                                                           target_classes=target_classes_array)
        logging.info("Stored woe representation of the features.")
        return transformed_features, target_classes_dict, encoder


def calculate_features(features_data, target_classes=None, encoder=None):
    """
    Process features data using WOE encoding. If target_classes is provided, it assumes
    the operation is for training data and creates an encoder. If an encoder is provided,
    it assumes the operation is for test data and uses the given encoder.

    Parameters:
    - features_data: pandas DataFrame, features data to be processed.
    - target_classes: pandas Series or numpy array, target variable for training data.
                      This should be None when processing test data.
    - encoder: WOEEncoder object, trained encoder for test data processing.
               This should be None when processing training data.

    Returns:
    - For training data: Tuple of (processed features_data, trained encoder)
    - For test data: processed features_data
    """
    try:
        # Validate input arguments for training or testing scenarios
        if target_classes is not None and encoder is not None:
            logging.error("Only target_classes for training or encoder for testing should be provided, not both.")
            raise ValueError("Only target_classes for training or encoder for testing should be provided, not both.")
        if target_classes is None and encoder is None:
            logging.error("Either target_classes for training or encoder for testing must be provided.")
            raise ValueError("Either target_classes for training or encoder for testing must be provided.")

        # Log the mode of operation based on the provided arguments
        mode = "training" if target_classes is not None else "testing"
        logging.info(f"Processing WOE features for {mode} data.")

        # Training data processing
        if target_classes is not None:
            encoder = WOEEncoder(retain_only_predictive_features=False, drop_invariant=True)
            features_encoded = encoder.fit_transform(features_data, target_classes)
            logging.info("Training data encoded successfully.")
        # Test data processing
        else:
            features_encoded = encoder.transform(features_data)
            logging.info("Test data encoded successfully.")

        # Update features_data with encoded features and apply datetime transformations
        features_data = features_data.copy()  # Avoid modifying original dataframe
        features_data.update(features_encoded)
        features_data = cast_dataframe_columns(features_data, features_encoded)
        features_data = keep_datetime_or_matched_columns(features_data, features_encoded)
        # features_data = features_data[features_encoded.columns.intersection(features_data.columns)]
        features_data = extract_datetime_features(features_data)

        return (features_data, encoder) if target_classes is not None else features_data

    except Exception as es:
        logging.exception(f"An error occurred: {es}")
        return None


def keep_datetime_or_matched_columns(df_original, df_reference):
    """
    Keep in df_original all datetime columns and columns that are present in df_reference.

    Parameters:
    - df_original: The DataFrame from which columns will be conditionally kept.
    - df_reference: The DataFrame used as reference for column names.

    Returns:
    - df_modified: The modified DataFrame with the specified columns kept.
    """
    # Identify datetime columns in df_original
    datetime_cols = [col for col in df_original.columns if pd.api.types.is_datetime64_any_dtype(df_original[col])]

    # Identify columns in df_original that are also in df_reference
    cols_in_reference = set(df_original.columns).intersection(set(df_reference.columns))

    # Combine the columns to keep: datetime columns + columns present in both DataFrames
    cols_to_keep = list(set(datetime_cols).union(cols_in_reference))

    # Select the columns to keep from df_original
    df_modified = df_original[cols_to_keep]

    return df_modified


def cast_dataframe_columns(df_to_cast, df_reference):
    """
    Cast columns of df_to_cast to have the same data type as corresponding columns in df_reference.

    Parameters:
    - df_to_cast: DataFrame whose columns you want to cast.
    - df_reference: DataFrame that serves as the reference for the data types.

    Returns:
    - DataFrame with casted columns.
    """
    for column in df_to_cast.columns:
        if column in df_reference.columns:
            # Cast column only if it exists in both DataFrames
            target_dtype = df_reference[column].dtype
            df_to_cast[column] = df_to_cast[column].astype(target_dtype)
    return df_to_cast


def extract_datetime_features(features_data: pd.DataFrame, prefix: str = 'default', drop_orig: bool = True):
    """
    Extracts datetime features from columns of datetime types in the given DataFrame.

    Parameters:
    - features_data: pd.DataFrame, DataFrame containing one or more datetime columns.
    - prefix: str, prefix for the newly created feature columns. If 'default', the original column name is used as prefix.
    - drop_orig: bool, whether to drop the original datetime columns.

    Returns:
    - pd.DataFrame with additional datetime features.
    """
    if not isinstance(features_data, pd.DataFrame):
        logging.error("Input is not a pandas DataFrame.")
        raise ValueError("Input must be a pandas DataFrame.")

    df = features_data.copy()
    datetime_types = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
    datetime_columns = df.select_dtypes(include=datetime_types).columns

    if not datetime_columns.any():
        logging.warning("No datetime columns found in the DataFrame.")
        return df

    new_columns = {}  # Dictionary to hold new columns before concatenation

    for var in datetime_columns:
        pfx = f"{var}:" if prefix == 'default' else prefix
        features = ['year', 'quarter', 'month', 'day', 'day_of_week', 'day_of_year',
                    'weekofyear', 'is_month_end', 'is_month_start', 'is_quarter_end',
                    'is_quarter_start', 'is_year_end', 'is_year_start', 'hour', 'minute', 'second']

        for feature in features:
            try:
                new_columns[f'{pfx}{feature}'] = getattr(df[var].dt, feature)
            except Exception as e:
                logging.error(f"Error extracting {feature} from {var}: {e}")

        # Additional feature: is_weekend
        new_columns[f'{pfx}is_weekend'] = np.where(df[var].dt.day_of_week.isin([5, 6]), 1, 0)

        # Concatenate all new columns at once
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        if drop_orig:
            df.drop(var, axis=1, inplace=True)
            logging.info(f"Dropped original column: {var}")

    logging.info("Datetime features extraction completed successfully.")
    return df
