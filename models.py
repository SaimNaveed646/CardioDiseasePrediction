from db import Base
from sqlalchemy import Column, Integer, String, Float, Boolean

class CardioVascular(Base):
    __tablename__ = "cardio"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    height = Column(Float)
    weight = Column(Float)
    ap_hi = Column(Integer)
    ap_lo = Column(Integer)
    cholesterol = Column(Integer)
    gluc = Column(Integer)
    smoke = Column(Boolean)
    alco = Column(Boolean)
    active = Column(Boolean)
    prediction = Column(Integer)  # add this