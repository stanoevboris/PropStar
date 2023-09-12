from sqlalchemy import engine
from conf import *
from abc import ABC, abstractmethod
import pandas as pd


def get_mssql_primary_keys_from_schema(target_schema: str):
    pass


def get_mssql_foreign_keys_from_schema(target_schema: str):
    pass


def create_mysql_connection(database: str):
    if not database:
        raise ValueError("Database missing!")

    return engine.URL.create(
        MYSQL_DBAPI,
        username=MYSQL_USER,
        password=MYSQL_PASS,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=database
    )


class Database(ABC):

    @abstractmethod
    def get_table(self, table_name: str, connection):
        pass

    @abstractmethod
    def get_primary_keys(self):
        pass

    @abstractmethod
    def get_foreign_keys(self):
        pass


class MSSQLDatabase(Database):

    def __init__(self, target_schema: str):
        self.username = MSSQL_USER
        self.password = MSSQL_PASS
        self.host = MSSQL_HOST
        self.port = MSSQL_PORT
        self.database = MSSQL_DEFAULT_DB
        self.query = MSSQL_QUERY

        self.target_schema = target_schema

        self.connection_url = engine.URL.create(
            MSSQL_DBAPI,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self.query
        )

    def get_table(self, table_name: str, connection):
        df = pd.read_sql(f"SELECT * FROM {self.target_schema}.{table_name}", connection)
        return df

    def get_primary_keys(self):
        return f"""SELECT
                        t.name AS TableName,
                        c.name AS PrimaryKeyColumn
                    FROM
                        sys.tables AS t
                    INNER JOIN
                        sys.indexes AS i
                    ON
                        t.object_id = i.object_id
                    INNER JOIN
                        sys.index_columns AS ic
                    ON
                        i.object_id = ic.object_id
                        AND i.index_id = ic.index_id
                    INNER JOIN
                        sys.columns AS c
                    ON
                        ic.object_id = c.object_id
                        AND ic.column_id = c.column_id
                    WHERE
                        t.schema_id = SCHEMA_ID('{self.target_schema}')
                        AND i.is_primary_key = 1;"""

    def get_foreign_keys(self):
        return f""" SELECT   
                        fk.name AS ForeignKeyName,
                        OBJECT_NAME(fkc.parent_object_id) AS ChildTable,
                        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ChildColumn,
                        OBJECT_NAME(fkc.referenced_object_id) AS ReferencedTable,
                        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
                    FROM
                        sys.foreign_keys AS fk
                    INNER JOIN
                        sys.foreign_key_columns AS fkc
                    ON
                        fk.object_id = fkc.constraint_object_id
                    WHERE
                        OBJECT_SCHEMA_NAME(fkc.parent_object_id) = '{self.target_schema}'
                        AND OBJECT_SCHEMA_NAME(fkc.referenced_object_id) = '{self.target_schema}';"""


class MYSQLDatabase(Database):

    def __init__(self, target_schema: str):
        self.username = MYSQL_USER
        self.password = MYSQL_PASS
        self.host = MYSQL_HOST
        self.port = MYSQL_PORT
        self.database = target_schema
        self.query = MSSQL_QUERY

        self.target_schema = target_schema

        if not self.database:
            raise ValueError("Database missing!")
        self.connection_url = engine.URL.create(
            MYSQL_DBAPI,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )

    def get_table(self, table_name: str, connection):
        df = pd.read_sql(f"SELECT * FROM {self.target_schema}.{table_name}", connection)
        return df

    def get_primary_keys(self):
        return f"""SELECT 
                        TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE 
                    FROM 
                        INFORMATION_SCHEMA.COLUMNS 
                    WHERE 
                        TABLE_SCHEMA = '{self.target_schema}' 
                        AND COLUMN_KEY = 'PRI' 
                    ORDER BY TABLE_NAME;"""

    def get_foreign_keys(self):
        return f"""SELECT
                        CONSTRAINT_NAME AS ForeignKeyName,
                        TABLE_NAME AS ChildTable,
                        COLUMN_NAME AS ChildColumn,
                        REFERENCED_TABLE_NAME AS ReferencedTable,
                        REFERENCED_COLUMN_NAME AS ReferencedColumn
                    FROM
                        information_schema.KEY_COLUMN_USAGE
                    WHERE
                        CONSTRAINT_SCHEMA = '{self.target_schema}'
                        AND REFERENCED_TABLE_NAME IS NOT NULL;"""
