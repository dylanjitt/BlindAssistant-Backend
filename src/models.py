from pydantic import BaseModel
from typing import List


class Billete(BaseModel):
  value:float
  position:List[List[int]]
