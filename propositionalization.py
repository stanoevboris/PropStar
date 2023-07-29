import numpy as np
import pandas as pd
import queue
import networkx as nx
import tqdm
from collections import defaultdict, OrderedDict
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import preprocessing
import re
from scipy import sparse

from neural import *  ## DRMs
from learning import *  ## starspace
from vectorizers import *  ## ConjunctVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from category_encoders import woe

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def table_generator(sql_file, variable_types):
    """
    A simple SQLite parser. This is inspired by the official SQL library, yet keeps only minimal overhead.
    input: a .sql data dump from e.g., relational.fit.cz
    output: Pandas represented-linked dataframe
    """

    table_trigger = False
    table_header = False
    current_table = None
    sqt = defaultdict(list)
    tabu = ["KEY", "PRIMARY", "CONSTRAINT"]
    table_keys = defaultdict(list)
    primary_keys = {}
    foreign_key_graph = []
    fill_table = False
    tables = dict()
    header_init = False
    col_types = []

    ## Read the file table-by-table (This could be done in a lazy manner if needed)
    with open(sql_file, "r", encoding="utf-8", errors="ignore") as sqf:
        for line in sqf:

            if "CREATE TABLE" in line:
                header_init = True

            if header_init:
                if "DEFAULT" in line:
                    if "ENGINE" in line:
                        continue

                    ctype = line.split()[1]
                    col_types.append(ctype)

            if "INSERT INTO" in line:

                ## Do some basic cleaning and create the dataframe
                table_header = False
                header_init = False
                vals = line.strip().split()
                vals_real = " ".join(vals[4:]).split("),(")
                vals_real[0] = vals_real[0].replace("(", "")
                vals_real[len(vals_real) - 1] = vals_real[len(vals_real) -
                                                          1].replace(");", "")
                col_num = len(sqt[current_table])

                vx = list(
                    filter(lambda x: len(x) == col_num, [
                        re.split(r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)", x)
                        for x in vals_real
                    ]))

                if len(vx) == 0:

                    ## this was added for the movies.sql case
                    vx = []

                    for x in vals_real:
                        parts = x.split(",")
                        vx.append(parts[len(parts) - col_num:])

                dfx = pd.DataFrame(vx)

                ## Discretize continuous attributes.
                #                if dfx.shape[1] == len(col_types):
                #                    dfx = discretize_candidates(dfx,col_types)

                col_types = []

                try:
                    assert dfx.shape[1] == len(sqt[current_table])

                except:
                    logging.info(sqt[current_table])
                    logging.info(
                        col_num,
                        re.split(r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)",
                                 vals_real[0]))

                try:
                    dfx.columns = [clear(x) for x in sqt[current_table]
                                   ]  ## some name reformatting.
                except:
                    dfx.columns = [x for x in sqt[current_table]
                                   ]  ## some name reformatting.

                tables[current_table] = dfx

            ## get the foreign key graph.
            if table_trigger and table_header:
                line = line.strip().split()
                if len(line) > 0:
                    if line[0] not in tabu:
                        if line[0] != "--":
                            if re.sub(r'\([^)]*\)', '',
                                      line[1]).lower() in variable_types:
                                sqt[current_table].append(clear(line[0]))
                    else:
                        if line[0] == "KEY":
                            table_keys[current_table].append(clear(line[2]))

                        if line[0] == "PRIMARY":
                            primary_keys[current_table] = cleanp(clear(
                                line[2]))
                            table_keys[current_table].append(clear(line[2]))

                        if line[0] == "CONSTRAINT":
                            ## Structure in the form of (t1 a1 t2 a2) is used.
                            foreign_key_quadruplet = [
                                clear(cleanp(x)) for x in
                                [current_table, line[4], line[6], line[7]]
                            ]
                            foreign_key_graph.append(foreign_key_quadruplet)

            if "CREATE TABLE" in line:
                table_trigger = True
                table_header = True
                current_table = clear(line.strip().split(" ")[2])

    return tables, foreign_key_graph, primary_keys


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
                              vectorizer=None,
                              vectorization_type="tfidf",
                              num_features=10000):
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
                witem = "-".join([target_table, column_name, row[i]])
                feature_vectors[index].append(witem)
                num_witems += 1
                total_witems.add(witem)

    logging.info("Traversing other tables ..")
    for core_fk in core_foreign_keys:  ## this is normaly a single key.
        bfs_traversal = dict(
            nx.bfs_successors(fk_graph, (target_table, core_fk)))

        ## Traverse the row space
        for index, row in tqdm.tqdm(core_table.iterrows(),
                                    total=core_table.shape[0]):

            current_depth = 0
            to_traverse = queue.Queue()
            to_traverse.put(target_table)  ## seed table
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

                    ## link and generate witems
                    first_table = tables[first_table_name]
                    second_table = tables[next_table_name]
                    if first_table_name == target_table:
                        key_to_compare = row[first_table_key]
                    elif first_table_name != target_table and current_depth == 2:
                        key_to_compare = None
                        for edge in fk_graph.edges():
                            if edge[0][0] == target_table and edge[1][0] == first_table_name:
                                key_to_compare = first_table[first_table[
                                                                 edge[1][1]] == row[edge[0]
                                [1]]][first_table_key]
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
            feature_vectors, target_classes.array, vectorizer, encoder, vectorization_type=vectorization_type)
        return matrix, target_classes
    else:
        matrix, vectorizer = relational_words_to_matrix(
            feature_vectors,
            relation_order,
            target_classes.array,
            encoder,
            vectorization_type,
            max_features=num_features)
        logging.info("Stored sparse representation of the witemsets.")
        return matrix, target_classes, vectorizer


def relational_words_to_matrix_with_vec(fw,
                                        target_classes,
                                        vectorizer,
                                        encoder,
                                        vectorization_type="tfidf"):
    """
    Just do the transformation. This is for proper cross-validation (on the test set)
    """

    docs = []
    if vectorization_type == "tfidf" or vectorization_type == "binary":
        for k, v in fw.items():
            docs.append(set(v))
        mtx = vectorizer.transform(docs)
    elif vectorization_type == 'woe':
        resulting_documents = []
        for v in fw.values():
            resulting_documents.append(" ".join(v))
        encoded_matrix, word_corpus = one_hot_encode_test_data(encoder, resulting_documents)
        X = pd.DataFrame(encoded_matrix, columns=word_corpus)
        woe_encoded_train = vectorizer.transform(X=X)
        mtx = sparse.csr_matrix(woe_encoded_train)
    else:
        for k, v in fw.items():
            docs.append(" ".join(v))
        mtx = vectorizer.transform(docs)

    return mtx


def relational_words_to_matrix(fw,
                               relation_order,
                               target_classes,
                               encoder,
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

    elif vectorization_type == 'woe':
        label_encoder = preprocessing.LabelEncoder()
        target_classes = label_encoder.fit_transform(target_classes)
        resulting_documents = []
        for v in fw.values():
            resulting_documents.append(" ".join(v))
        encoded_matrix, word_corpus = one_hot_encode_train_data(encoder, resulting_documents)
        X = pd.DataFrame(encoded_matrix, columns=word_corpus)
        vectorizer = woe.WOEEncoder(cols=word_corpus)
        woe_encoded_train = vectorizer.fit_transform(X=X, y=target_classes)
        mtx = sparse.csr_matrix(woe_encoded_train)

    return mtx, vectorizer


def one_hot_encode_train_data(encoder, word_lists):
    encoded = encoder.fit_transform(word_lists)
    labels = sorted(encoder.vocabulary_.keys())
    return encoded.toarray(), labels


def one_hot_encode_test_data(encoder, word_lists):
    encoded = encoder.transform(word_lists)
    labels = sorted(encoder.vocabulary_.keys())
    return encoded.toarray(), labels
