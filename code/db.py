import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()

pool: asyncpg.pool.Pool | None = None


async def init_db_pool():
    global pool
    pool = await asyncpg.create_pool(dsn=os.environ["NEON_DB_DSN"])
    print("Database connection pool created.")


async def close_db_pool():
    global pool
    if pool:
        await pool.close()
        print("Database connection pool closed.")
