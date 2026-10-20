from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings


# 1. Configure database connections
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)


# 2. Define the base class for ORM models
class Base(DeclarativeBase):
    pass


# 3. Create a factory for database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# 4. Provide one session per dependency lifecycle
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
