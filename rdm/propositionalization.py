import itertools
from abc import ABC, abstractmethod
import networkx as nx
import pandas as pd
from tqdm import tqdm
import logging
import queue

from rdm.utils import OrderedDictList

# TODO: implement feature engineering and feature selection
# TODO: include flexibility for custom steps... this can be included maybe in new class

from typing import Dict, Optional, List


# Base configuration class to simplify initializers
class PropConfig:
    def __init__(self, tables: Dict, foreign_keys: Dict, primary_keys: Optional[Dict] = None,
                 target_table: Optional[str] = None, target_attribute: Optional[str] = None, max_depth: int = 2):
        self.tables = tables
        self.target_table = target_table
        self.target_attribute = target_attribute
        self.foreign_keys = foreign_keys
        self.primary_keys = primary_keys

        self.core_table = self.tables[target_table] if target_table else None
        self.target_classes = self.core_table[target_attribute] if target_table else None

        self.core_foreign_keys = set()
        self.all_foreign_keys = set()

        self.max_depth = max_depth


class Propositionalization(ABC):
    def __init__(self, config: PropConfig):
        self.config = config
        self.initialize_logging()
        self.fk_graph = self.create_fk_graph(self.config.foreign_keys)
        self.feature_vectors = OrderedDictList()
        self.total_witems = set()

    @staticmethod
    def initialize_logging():
        logging.basicConfig(
            format='[%(asctime)s] p%(process)s {%(pathname)s:%(funcName)s} %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S',
            level=logging.INFO)

    def create_fk_graph(self, foreign_keys):
        graph = nx.Graph()
        for t1, k1, t2, k2 in foreign_keys:
            if t1 == self.config.target_table:
                self.config.core_foreign_keys.add(k1)

            elif t2 == self.config.target_table:
                self.config.core_foreign_keys.add(k2)

            self.config.all_foreign_keys.add(k1)
            self.config.all_foreign_keys.add(k2)
            graph.add_edge(t1, t2, source_column=k1, target_column=k2)
        return graph

    def print_graph(self):
        logging.info("Graph Representation of Foreign Key Relationships:")
        logging.info("\nNodes (Tables):")
        for node in self.fk_graph.nodes():
            print(node)

        logging.info("\nEdges (Foreign Key Relationships):")
        for t1, t2, attributes in self.fk_graph.edges(data=True):
            logging.info(
                f"From {t1} to {t2} - Source Column: {attributes['source_column']}, "
                f"Target Column: {attributes['target_column']}")

    def initialize_queue(self, traversal_map):
        """Initializes the queue with successor tables of the target table."""
        to_traverse = queue.Queue()
        successor_tables = traversal_map.get(self.config.target_table, [])
        # logging.info(f"Successor Tables: {successor_tables}")
        for source_table in successor_tables:
            to_traverse.put((self.config.target_table, 1, source_table))  # queue stores tuples of (table name, depth)
        return to_traverse

    def fill_queue(self, current_table, current_depth, traversal_map, to_traverse):
        """Utility function to fill the queue based on the current table and depth."""
        if current_depth < self.config.max_depth:
            future_tables = traversal_map.get(current_table, [])
            for next_table in future_tables:
                to_traverse.put((current_table, current_depth + 1, next_table))

            # logging.info(f"Queue State: {list(to_traverse.queue)}")
            # logging.info(f"Future tables from {current_table}: {future_tables}")
        return to_traverse

    @abstractmethod
    def traverse_and_fetch_related_data(self):
        pass

    @abstractmethod
    def run(self):
        pass


class Wordfication(Propositionalization):
    def __init__(self, config: PropConfig):
        super().__init__(config)

        self.docs = []

    def propositionalize_core_table(self):
        logging.info("Propositionalization of core table...")
        excluded_columns = self.config.core_foreign_keys.union({self.config.target_attribute})

        columns = self.config.core_table.columns.difference(excluded_columns)
        for row in tqdm(self.config.core_table.itertuples(index=True), total=self.config.core_table.shape[0]):
            for column in columns:
                value = getattr(row, column)
                witem = f"{self.config.target_table}-{column}-{str(value)}"
                self.feature_vectors[row.Index].append(witem)
                self.total_witems.add(witem)

    def traverse_and_fetch_related_data(self):
        logging.info("Traversing other tables...")
        traversal_map = dict(nx.bfs_successors(self.fk_graph, self.config.target_table))

        for row in tqdm(self.config.core_table.itertuples(index=True),
                        total=self.config.core_table.shape[0]):
            parsed_tables = {self.config.target_table}  # to avoid future circular join to the target table
            to_traverse = self.initialize_queue(traversal_map=traversal_map)
            while not to_traverse.empty():
                parent_table, current_depth, current_table = to_traverse.get()

                if current_table not in parsed_tables:
                    parsed_tables.add(current_table)
                    # logging.info(f"Currently applying wordification over table: {current_table} "
                    #              f"at depth {current_depth}")
                    edge_data = self.fk_graph.get_edge_data(parent_table, current_table)
                    source_column, target_column = edge_data['source_column'], edge_data['target_column']
                    if source_column not in self.config.core_table:
                        source_column, target_column = target_column, source_column
                    next_table_df = self.config.tables[current_table]
                    column_value_to_be_searched = getattr(row, source_column)
                    table_row = next_table_df[next_table_df[target_column] == column_value_to_be_searched]
                    table_row.reset_index(drop=True, inplace=True)

                    if table_row.empty:
                        continue

                    excluded_columns = self.config.all_foreign_keys.union({self.config.target_attribute})
                    for column_name in table_row:
                        value = [value for value in table_row[column_name]][0]
                        if column_name not in excluded_columns:
                            witem = f"{current_table}-{column_name}-{value}"
                            self.total_witems.add(witem)
                            self.feature_vectors[row.Index].append(witem)

                    to_traverse = self.fill_queue(current_table=current_table,
                                                  current_depth=current_depth,
                                                  traversal_map=traversal_map,
                                                  to_traverse=to_traverse)

    def features2docs(self):
        self.docs = [' '.join(value) for value in self.feature_vectors.values()]

    def run(self) -> [List, pd.Series]:
        self.propositionalize_core_table()
        self.traverse_and_fetch_related_data()
        self.features2docs()
        return self.docs, self.config.target_classes


class Denormalization(Propositionalization):
    def __init__(self, config: PropConfig):
        super().__init__(config)

    def traverse_and_fetch_related_data(self) -> pd.DataFrame:
        logging.info("Traversing other tables...")
        traversal_map = dict(nx.bfs_successors(self.fk_graph, self.config.target_table))
        features_data = self.config.core_table.copy()

        parsed_tables = {self.config.target_table}  # to avoid future circular join to the target table
        to_traverse = self.initialize_queue(traversal_map=traversal_map)

        while not to_traverse.empty():
            parent_table, current_depth, current_table = to_traverse.get()

            if current_table not in parsed_tables:
                parsed_tables.add(current_table)
                logging.info(f"Currently applying denormalization over table: {current_table} "
                             f"at depth {current_depth}")
                edge_data = self.fk_graph.get_edge_data(parent_table, current_table)
                source_column, target_column = edge_data['source_column'], edge_data['target_column']
                if source_column not in features_data or source_column in self.config.tables[current_table]:
                    source_column, target_column = target_column, source_column
                features_data = features_data.merge(self.config.tables[current_table],
                                                    how='inner',
                                                    left_on=source_column,
                                                    right_on=target_column,
                                                    suffixes=(None, f'__{current_table}'))

                # Extract keys from foreign and primary keys, excluding core foreign keys
                all_keys = set(itertools.chain(self.config.all_foreign_keys, self.config.primary_keys.values()))
                excluded_keys = all_keys - self.config.core_foreign_keys

                # Append '__y' suffix and filter by presence in features_data.columns
                columns_to_drop = {f"{key}__{current_table}" for key in excluded_keys if f"{key}__{current_table}" in features_data.columns}

                features_data.drop(list(columns_to_drop), axis=1, inplace=True)

                to_traverse = self.fill_queue(current_table=current_table,
                                              current_depth=current_depth,
                                              traversal_map=traversal_map,
                                              to_traverse=to_traverse)

        return features_data

    def clear_columns(self, features: pd.DataFrame):
        """
        Method that will drop all columns related to primary key or foreign key
        """
        available_keys = {key for key in self.config.all_foreign_keys.union(self.config.primary_keys.values())
                          if key in features.columns}

        # cols_to_drop = [col for col in features.columns if col in available_keys and col.endswith('__y')]
        cols_to_drop = [col for col in available_keys if col in features.columns]
        features.drop(cols_to_drop, axis=1, inplace=True)

        return features

    def prepare_labels(self, features: pd.DataFrame) -> None:
        self.config.target_classes = features[self.config.target_attribute]
        features.drop(self.config.target_attribute, axis=1, inplace=True)

    def run(self) -> [pd.DataFrame, pd.Series]:
        features_data = self.traverse_and_fetch_related_data()
        features_data = self.clear_columns(features_data)

        self.prepare_labels(features_data)
        return features_data, self.config.target_classes
