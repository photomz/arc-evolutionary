import asyncio
import os
from src.db import init_db_pool, close_db_pool, pool


async def test_connection():
    await init_db_pool()
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT version()")
        print(f"Connected to: {version}")
    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(test_connection())
