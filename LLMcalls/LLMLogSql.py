import sqlite3
import threading

# LLMLogSql用于记录调用日志
class LLMLogSql:
    """
    def __init__(self, log_file) -> None:
        # log_file日志文件路径名
        self.log_file = log_file
        conn = sqlite3.connect(self.log_file) # 创建一个到sql数据库的链接
        cursor = conn.cursor() # 创建用于数据库操作的光标
        cursor.execute(
            ""CREATE TABLE IF NOT EXISTS my_table
            (Q TEXT PRIMARY KEY, V TEXT)""(这里原来有三个引号)
        ) # my_table用于存储调用日志，Q是主键，V是值
        self.lock = threading.Lock() # 线程锁，用于防止多个线程同时写入数据库
        # 创建线程局部数据对象
        # 线程局部数据是只有在当前线程中可见的数据，
        # 可以用来存储每个线程的特定状态，而不会与其他线程发生冲突
        self.local = threading.local()

    def get_connection(self):
        # 获取线程本地的数据库连接，如果不存在则创建新连接,并存储在线程本地变量之中
        # 每个线程都维护自己的数据库链接，避免多线程环境下出现连接冲突
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.log_file)
        return self.local.conn
    """

    def __init__(self, log_file) -> None:
        self.log_file = log_file
        self.lock = threading.Lock()
        self.local = threading.local()
        # 在初始化时安全创建表并关闭连接
        self.init_db()

    def init_db(self):
        # 创建或确认数据库表存在，并初始化数据库
        conn = sqlite3.connect(self.log_file)  # 创建一个到sql数据库的链接
        cursor = conn.cursor()  # 创建用于数据库操作的光标
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS my_table
            (Q TEXT PRIMARY KEY, V TEXT)"""
        )
        conn.commit()  # 提交创建表的操作
        cursor.close()  # 关闭光标
        conn.close()  # 关闭连接


    def DBQuery(self, Q):
        # 从数据库中查询键Q对应的值
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT V FROM my_table WHERE Q=?", (Q,))
            # fetchone()方法从查询结果中获取下一行，如果没有更多行，则返回None。
            # 在这种情况下，它会返回my_table表中Q对应的V值，如果找到的话。
            result = cursor.fetchone()
        return result[0] if result else None

    """
    def get_connection(self):
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.log_file)
        return self.local.conn
    """
    def get_connection(self):
        # 检查线程局部存储中是否已有连接，如果有则验证连接是否仍然开放
        if hasattr(self.local, "conn"):
            try:
                # 尝试在连接上执行一个简单的无操作命令来验证连接是否活跃
                self.local.conn.execute("SELECT 1")
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                # 如果连接失效，关闭后重新建立连接
                self.local.conn.close()
                self.local.conn = sqlite3.connect(self.log_file)
        else:
            # 如果线程中没有连接，创建一个新的连接
            self.local.conn = sqlite3.connect(self.log_file)
        return self.local.conn

    def DBInsert(self, Q, V):
        # 向数据库中插入对应键值对(Q, V)
        with self.lock:
            conn = self.get_connection() # REVISE
            cursor = conn.cursor()
            # 查询数据库中是否有包含键Q的键值对
            # 如果有，更新Q对应的值为V
            # 如果没有，插入(Q, V)
            cursor.execute(
                "INSERT OR REPLACE INTO my_table (Q, V) VALUES (?, ?)", (Q, V)
            )
            conn.commit()
