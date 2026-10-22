from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://fantasy_26:fantasy_26@localhost:5432/fantasy_26"
    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    entry_id: int = 0  # FPL team id

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3(cls, v: str) -> str:
        """Neon (and most hosts) hand out `postgresql://` or `postgres://` URLs.
        SQLAlchemy reads those as psycopg2, which isn't installed."""
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    class Config:
        env_file = ".env"
        # .env also holds values only scripts read (e.g. NEON_DATABASE_URL for
        # scripts/copy_to_neon.sh); don't fail on keys that aren't settings.
        extra = "ignore"


settings = Settings()
