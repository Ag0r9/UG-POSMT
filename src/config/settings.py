from pydantic_settings import BaseSettings, SettingsConfigDict

from enum import Enum


class MaskingMethod(str, Enum):
    none = "none"
    shallow = "shallow"
    deep = "deep"
    only_children = "only_children"
    only_parent = "only_parent"
    only_self = "only_self"
    full = "full"


class Settings(BaseSettings):
    input_lang: str = "en"
    output_lang: str = "de"
    mt_model_name: str = "Helsinki-NLP/opus-mt-en-de"
    sample: bool = True
    masking_method: MaskingMethod = MaskingMethod.none

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
