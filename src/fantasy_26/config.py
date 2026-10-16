from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://fantasy_26:fantasy_26@localhost:5432/fantasy_26"
    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    entry_id: int = 0          # your FPL team id
    class Config:
        env_file = ".env"

settings = Settings()