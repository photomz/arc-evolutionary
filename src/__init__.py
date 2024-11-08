import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LOGFIRE_TOKEN"):
    os.environ["LOGFIRE_TOKEN"] = "y0hVmLsPlMSDSbQlGH2NB1lQ71DblNkM5p6lnB0VrZdY"

if not os.getenv("NEON_DB_DSN"):
    os.environ["NEON_DB_DSN"] = (
        "postgresql://kaggle:mQh7DRLvVX4z@ep-dawn-dawn-a4zd48ba-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
    )
if not os.getenv("PLOT") or os.environ["PLOT"] == "0":
    PLOT = False
else:
    PLOT = True

import logfire

logfire.configure()
