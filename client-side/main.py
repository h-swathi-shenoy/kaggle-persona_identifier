import streamlit as st
from src import questionnaire, top_persona_view


def main():
    st.title("Kaggle Persona Identifier")
    menu = ["Know your Persona", "Look at top Personas"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Know your Persona":
        questionnaire.run_questionnaire()
    elif choice == "Look at top Personas":
        top_persona_view.run_show_persona()


if __name__ == '__main__':
    main()
