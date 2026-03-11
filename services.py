from sqlalchemy.orm import Session
import models
from schemas import UserInput


def save_user_input(db: Session, data: UserInput, prediction: int, name: str = "User"):
    new_entry = models.CardioVascular(
        name=name,
        age=data.age,
        gender=data.gender,
        height=data.height,
        weight=data.weight,
        ap_hi=data.ap_hi,
        ap_lo=data.ap_lo,
        cholesterol=data.cholesterol,
        gluc=data.gluc,
        smoke=data.smoke,
        alco=data.alco,
        active=data.active,
        prediction=prediction   # <-- add this
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry