import sqlite3
import time


class SQLWrapper:

    def __init__(self, dbname):
        self.dbname = dbname

    def __enter__(self):
        self.sqlite_connection = sqlite3.connect(self.dbname)
        self.cursor = self.sqlite_connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self.sqlite_connection.commit()
        self.cursor.close()
        self.sqlite_connection.close()


class SqlWorker:
    dbname = ""

    def __init__(self, dbname):
        self.dbname = dbname

        sqlite_connection = sqlite3.connect(dbname)
        cursor = sqlite_connection.cursor()
        cursor.execute(f"""CREATE TABLE if not exists chats (
                                    context TEXT NOT NULL PRIMARY KEY,
                                    dialog_text TEXT NOT NULL,
                                    memory_dump TEXT,
                                    first_use INTEGER NOT NULL DEFAULT {int(time.time())});""")
        sqlite_connection.commit()
        cursor.close()
        sqlite_connection.close()

    def dialog_update(self, context, dialog_text):
        with SQLWrapper(self.dbname) as sql_wrapper:
            sql_wrapper.cursor.execute("""SELECT * FROM chats WHERE context = ?""", (context,))
            record = sql_wrapper.cursor.fetchall()
            if not record:
                sql_wrapper.cursor.execute("""INSERT INTO chats VALUES (?,?,?,?);""",
                                           (context, dialog_text, None, int(time.time())))
            else:
                sql_wrapper.cursor.execute("""UPDATE chats SET dialog_text = ? WHERE context = ?""",
                                           (dialog_text, context))

    def memory_update(self, context, memory_dump):
        with SQLWrapper(self.dbname) as sql_wrapper:
            sql_wrapper.cursor.execute("""UPDATE chats SET memory_dump = ? WHERE context = ?""",
                                       (memory_dump, context))

    def dialog_get(self, context):
        with SQLWrapper(self.dbname) as sql_wrapper:
            sql_wrapper.cursor.execute("""SELECT * FROM chats WHERE context = ?""", (context,))
            dialog = sql_wrapper.cursor.fetchall()
            return dialog
