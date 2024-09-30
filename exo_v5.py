from pydantic import BaseModel
from datetime import datetime
import time

class Model(BaseModel):
    d: datetime = datetime.now()

o1 = Model()
print(o1.d)
time.sleep(1) # Wait for a second
o2 = Model()
print(o2.d)
print(o1.d < o2.d) # False