from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    DOCUMENTS_PATH: str = "clean_documents.txt"
    EMBEDDINGS_PATH: str = "models/product_embeddings.npy"
    INDEX_PATH: str = "models/product_index.faiss"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
