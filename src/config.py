from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache
# from dotenv import load_dotenv

# print(load_dotenv())

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    logMoney_file: str ="money.json"
    logSign_file:str="letreros.json"
    api_name: str = "Blind Assistant"
    revision: str = "local"
    model_moneyDet: str = "models/bolivian_money_detector_MK_I.pt"
    model_minibusSign:str= "models/minibus_sign_detector_MK_I.pt"
    #llm: str = 'llava'
    #llm: str = 'llama3.2-vision'#OPTION 3
    #llm: str = 'gemma3:4b'
    llm: str = 'gemma3:12b' #OPTION 2
    #llm: str = 'qwen2.5vl:7b'#OPTION 1
    
    log_level: str = "DEBUG"


@cache
def get_settings():
    print("getting settings...")
    return Settings()