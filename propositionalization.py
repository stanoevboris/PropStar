import itertools
from typing import Optional

import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import re

from neural import *  ## DRMs
from learning import *  ## starspace
from utils import clear, cleanp, OrderedDictList
from vectorizers import *  ## ConjunctVectorizer
from category_encoders import woe

import logging
import sqlalchemy as sa
from db_utils import MSSQLDatabase, MYSQLDatabase

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

sql_type_dict = {
    "mssql": MSSQLDatabase,
    "mysql": MYSQLDatabase
}


def get_data(sql_type: str, database: Optional[str], target_schema: str, include_all_schemas: Optional[bool]):
    db_object = sql_type_dict[sql_type]

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
                              vectorizer=None,
                              vectorization_type="tfidf",
                              num_features=10000,
                              primary_keys=None):
    """
    Key method for generation of relational words and documents.
    It traverses individual tables in path, and consequantially appends the witems to a witem set. This method is a rewritten, non exponential (in space) version of the original Wordification algorithm (Perovsek et al, 2014).
    input: a collection of tables and a foreign key graph
    output: a representation in form of a sparse matrix.
    """

    fk_graph = nx.Graph(
    )  ## a simple undirected graph as the underlying fk structure
    core_foreign_keys = set()
    all_foreign_keys = set()

    for foreign_key in fkg:

        ## foreing key mapping
        t1, k1, t2, k2 = foreign_key

        if t1 == target_table:
            core_foreign_keys.add(k1)

        elif t2 == target_table:
            core_foreign_keys.add(k2)

        all_foreign_keys.add(k1)
        all_foreign_keys.add(k2)

        ## add link, note that this is in fact a typed graph now
        fk_graph.add_edge((t1, k1), (t2, k2))

    ## this is more efficient than just orderedDict object
    feature_vectors = OrderedDictList()
    if not indices is None:
        core_table = tables[target_table].iloc[indices, :]
    else:
        core_table = tables[target_table]
    all_table_keys = get_table_keys(fkg)
    core_foreign = None
    target_classes = core_table[target_attribute]

    ## This is a remnant of one of the experiment, left here for historical reasons :)
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

    ## The main propositionalization routine
    logging.info("Propositionalization of core table ..")
    for index, row in tqdm.tqdm(core_table.iterrows(),
                                total=core_table.shape[0]):
        for i in range(len(row)):
            column_name = row.index[i]
            if column_name != target_attribute and not column_name in core_foreign_keys:
                witem = "-".join([target_table, column_name, str(row[i])])
                feature_vectors[index].append(witem)
                num_witems += 1
                total_witems.add(witem)

    logging.info("Traversing other tables ..")
    for core_fk in core_foreign_keys:  # this is normaly a single key.
        bfs_traversal = dict(
            nx.bfs_successors(fk_graph, (target_table, core_fk)))

        # Traverse the row space
        for index, row in tqdm.tqdm(core_table.iterrows(),
                                    total=core_table.shape[0]):

            current_depth = 0
            to_traverse = queue.Queue()
            to_traverse.put(target_table)  # seed table
            max_depth = 2
            tables_considered = 0
            parsed_tables = set()

            ## Perform simple search
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
                    if not first_table_name in tables or not next_table_name in tables:
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

    # Vectorizer is an arbitrary vectorizer, some of the well known ones are implemented here, it's simple to add your own!
    if vectorizer:
        matrix = relational_words_to_matrix_with_vec(
            feature_vectors, vectorizer, vectorization_type=vectorization_type)
        return matrix, target_classes.array
    else:
        matrix, vectorizer = relational_words_to_matrix(
            feature_vectors,
            relation_order,
            vectorization_type,
            max_features=num_features)
        logging.info("Stored sparse representation of the witemsets.")
        return matrix, target_classes.array, vectorizer


def relational_words_to_matrix_with_vec(fw,
                                        vectorizer,
                                        vectorization_type="tfidf"):
    """
    Just do the transformation. This is for proper cross-validation (on the test set)
    """

    docs = []
    if vectorization_type == "tfidf" or vectorization_type == "binary":
        for k, v in fw.items():
            docs.append(set(v))
        mtx = vectorizer.transform(docs)
    else:
        for k, v in fw.items():
            docs.append(" ".join(v))
        mtx = vectorizer.transform(docs)

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
                                     vectorizer=None,
                                     vectorization_type="tfidf",
                                     num_features=10000,
                                     primary_keys=None):
    """
    Key method for generation of relational words and documents.
    It traverses individual tables in path, and consequantially appends the witems to a witem set. This method is a rewritten, non exponential (in space) version of the original Wordification algorithm (Perovsek et al, 2014).
    input: a collection of tables and a foreign key graph
    output: a representation in form of a sparse matrix.
    """

    fk_graph = nx.Graph(
    )  # a simple undirected graph as the underlying fk structure
    core_foreign_keys = set()
    all_foreign_keys = set()

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

    target_classes = core_table[target_attribute]
    try:
        target_classes_dict = dict(zip(core_table[primary_keys[target_table]].astype('int32'),
                                       core_table[target_attribute]))
    except Exception as es:
        print(es)
        target_classes_dict = dict(zip(core_table[primary_keys[target_table]], core_table[target_attribute]))

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
                if (table) in parsed_tables:
                    continue
                parsed_tables.add(table)
                first_table_name, first_table_key = origin, core_fk
                next_table_name, next_table_key = table
                if not first_table_name in tables or not next_table_name in tables:
                    continue

                # link and generate witems
                second_table = tables[next_table_name]

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
    features_data.drop(target_attribute, axis=1, inplace=True)
    features_data = features_data.loc[:, ~features_data.columns.str.endswith('__y')]
    features_data = features_data.apply(pd.to_numeric, errors='ignore')
    try:
        features_data.set_index(primary_keys[target_table], inplace=True)
    except Exception as es:
        print(es)
    target_classes_array = [target_classes_dict[index] for index in features_data.index]
    if vectorizer:
        matrix = calculate_woe_test_data(features_data=features_data,
                                         vectorizer=vectorizer)
        logging.info("Stored sparse representation of the witemsets.")

        return matrix, target_classes_dict
    else:
        matrix, vectorizer = calculate_woe_train_data(features_data=features_data, target_classes=target_classes_array)
        logging.info("Stored sparse representation of the witemsets.")
        return matrix, target_classes_dict, vectorizer


def calculate_woe_train_data(features_data, target_classes):
    string_cols = features_data.select_dtypes(include='object').columns.values
    woe_data = features_data[string_cols].copy()

    vectorizer = woe.WOEEncoder(cols=string_cols)

    try:
        label_encoder = preprocessing.LabelEncoder()
        target_classes = label_encoder.fit_transform(target_classes)
        woe_encoded = vectorizer.fit_transform(X=woe_data, y=target_classes)
        features_data.update(woe_encoded)
        return features_data, vectorizer

    except Exception as es:
        print(es)
        return


def calculate_woe_test_data(features_data, vectorizer):
    string_cols = features_data.select_dtypes(include='object').columns.values
    woe_data = features_data[string_cols].copy()

    woe_encoded = vectorizer.transform(X=woe_data)
    features_data.update(woe_encoded)
    return features_data
