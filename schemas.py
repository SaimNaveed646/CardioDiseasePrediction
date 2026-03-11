from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated

class UserInput(BaseModel):
    name: str = Field(..., min_length=1, description="Full name of the user")  # mandatory
    age: Annotated[int, Field(..., gt=0, lt=120)]
    gender: Annotated[Literal["male", "female"], Field(...)]
    height: Annotated[float, Field(..., gt=0, lt=2.5)]
    weight: Annotated[float, Field(..., gt=0, lt=300)]
    ap_hi: Annotated[int, Field(..., gt=50, lt=300)]
    ap_lo: Annotated[int, Field(..., gt=30, lt=200)]
    cholesterol: Annotated[Literal[1, 2, 3], Field(...)]
    gluc: Annotated[Literal[1, 2, 3], Field(...)]
    smoke: Annotated[bool, Field(...)]
    alco: Annotated[bool, Field(...)]
    active: Annotated[bool, Field(...)]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)
    
    @computed_field
    @property
    def genderr(self) -> int:
        return 1 if self.gender == "male" else 2
    
    @computed_field
    @property
    def bmi_category(self) -> str:
        if self.bmi < 18.5:
            return "Underweight"
        elif self.bmi < 25:
            return "Normal"
        elif self.bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 45:
            return "adult"
        elif self.age < 60:
            return "Middle_aged"
        return "Senior"