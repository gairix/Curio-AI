from typing import List
from pydantic import BaseModel, Field

class RAGResponse(BaseModel): 
    answer: str = Field(description="Answer generated from context")

class QuizQuestion(BaseModel):
    question: str = Field(description="The multiple choice question prompt text")
    options: List[str] = Field(description="Exactly 4 clean string answers options to choose from")
    correct_answer: str = Field(description="The exact match string identical to the correct option choice")
    explanation: str = Field(description="High clarity overview explaining why this selection is accurate")

class QuizSchema(BaseModel): 
    quiz: List[QuizQuestion] = Field(description="An explicit collection of 3-5 high analytical study questions")
