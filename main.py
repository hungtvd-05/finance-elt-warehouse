from psycopg2 import pool

# Tạo connection pool
connection_pool = None

try:
    connection_pool = pool.SimpleConnectionPool(
        minconn=5,
        maxconn=10,
        host="localhost",
        port="5432",
        database="dev",
        user="db_user",
        password="db_password"
    )

    if connection_pool:
        print("✅ Kết nối database thành công!")

        # Lấy 1 connection từ pool để test
        conn = connection_pool.getconn()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()
            print(f"📌 PostgreSQL version: {db_version[0]}")

            # Liệt kê tất cả các tables trong database
            cursor.execute("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """)
            tables = cursor.fetchall()

            if tables:
                print(f"\n📋 Danh sách tables trong database 'dev':")
                print("-" * 50)
                for schema, table in tables:
                    print(f"   📁 {schema}.{table}")
            else:
                print("\n⚠️ Database trống - chưa có table nào!")

            cursor.close()
            connection_pool.putconn(conn)

except Exception as e:
    print(f"❌ Lỗi kết nối database: {e}")

finally:
    # Đóng tất cả connections khi không dùng nữa
    if connection_pool:
        connection_pool.closeall()
        print("🔒 Đã đóng connection pool")

