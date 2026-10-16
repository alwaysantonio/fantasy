from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fantasy_26.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,     # survives Postgres dropping idle connections
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)