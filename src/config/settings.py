from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    input_lang: str = "en"
    output_lang: str = "de"
    mt_model_name: str = "Helsinki-NLP/opus-mt-en-de"
    sample: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
