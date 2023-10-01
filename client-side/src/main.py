import streamlit as st
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException,Body
import uvicorn
from questionnaire import questionnaire
from top_persona_view import show_persona

app = FastAPI()

@app.get("/get")
async def get():
    return {"status": "OK"}


def main():
    st.title("Kaggle Persona Identifier")
    menu = ["Know your Persona", "Look at top Personas"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Know your Persona":
        questionnaire()
    elif choice == "Look at top Personas":
        show_persona()






if __name__ == '__main__':
    main()
