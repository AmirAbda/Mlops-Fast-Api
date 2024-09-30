from pydantic import BaseModel

"""
Up to now , we've assumed that each field had to be provided when 
instantiating the model 
"""
class UserProfile(BaseModel):
    nickname: str
    location: str | None = None
    subscribed_newsletter: bool = False