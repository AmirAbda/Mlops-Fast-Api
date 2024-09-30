from pydantic import BaseModel

class Address(BaseModel):
    street_address:str
    postal_code: str
    city: str
    country: str
class Person(BaseModel):
    first_name:str
    last_name:str
    age:int 
    addrress:Address

