import json
import yaml
import pandas as pd
import streamlit as st
import numpy as np
import random
from pathfinder import PathConfig

path_obj = PathConfig()
data_dir = path_obj.data_dir
configs = path_obj.configs_dir


def boolean_single_select(onehot: pd.DataFrame) -> str:
    print(onehot.shape)
    col_shape = onehot.shape[1]
    for i in range(col_shape):
        if onehot.iloc[:, i].any() == 1:
            return onehot.iloc[:, i]
    return


def boolean_multi_select(onehot: pd.DataFrame, col_label : str) -> str:
    print(onehot.shape)
    col_shape = onehot.shape[1]
    selected_options = []
    for i in range(col_shape):
        if onehot.iloc[:, i].any() == 1:
            selected_options.append(onehot.iloc[:,i].name)
    if len(selected_options)!=0:
        if '-' in selected_options[0]:
            selected_options = [i.split('-')[2] for i in selected_options]
        elif '_'  in selected_options:
            selected_options = [i.split('_')[2] for i in selected_options]
        else:
            selected_options = selected_options
    if len(selected_options) ==0 and 'Language' in col_label:
            selected_options = ['Python', 'JavaScript']
    if len(selected_options) ==0 and 'IDE' in col_label:
            selected_options = [' Jupyter (JupyterLab, Jupyter Notebooks, etc) ', '  PyCharm ']

    return  selected_options


def persona_card(persona:dict) -> None:
    with st.chat_message(name='Q1', avatar='🧑‍💻'):
        st.write(f"**1**. What is the highest level of formal education that you have attained/or plan to attain in next two years?  \n **{persona['education'].name}**")
    with st.chat_message(name='Q2', avatar='🧑‍💻'):
        st.write(f"**2**.  For how many years have you been writing code and/or programming?  \n **{persona['coding']}**")
    with st.chat_message(name='Q3', avatar='🧑‍💻'):
        st.write(f"**3**.  What programming languages do you use on a regular basis? (Select all that apply)  \n **{persona['language']}**")
    with st.chat_message(name='Q4', avatar='🧑‍💻'):
        st.write(f"**4**. Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply)  \n **{persona['ide']}**")
    with st.chat_message(name='Q5', avatar='🧑‍💻'):
        st.write(f"**5**. Count of known ML Frameworks : \t **{int(persona['ml_framework'])}**")
    with st.chat_message(name='Q6', avatar='🧑‍💻'):
        st.write(f"**6**. Count of known ML Algorithms : \t **{int(persona['ml_algo'])}**")
    with st.chat_message(name='Q7', avatar='🧑‍💻'):
        st.write(f"**7**. Count of known CV Algorithms : \t **{int(persona['cv_count'])}**")
    with st.chat_message(name='Q8', avatar='🧑‍💻'):
        st.write(f"**8**. Count of known NLP Algorithms : \t **{int(persona['nlp_count'])}**")
    with st.chat_message(name='Q9', avatar='🧑‍💻'):
        st.write(f"**9**. Count of known AutoML Algorithms : \t **{int(persona['automl_count'])}**")
    with st.chat_message(name='Q10', avatar='🧑‍💻'):
        st.write(f"**10**. Count of known Business Intelligence tools : \t **{int(persona['business_intel'])}**")
    with st.chat_message(name='Q11', avatar='🧑‍💻'):
        st.write(f"**11**. Count of known Business Data Products tools : \t **{int(persona['big_data_products'])}**")
    with st.chat_message(name='Q12', avatar='🧑‍💻'):
        st.write(f"**12**. Have you managed ML products on a regular basis? (Select all that apply) : \t **{int(persona['managed_ml'])}**")
    with st.chat_message(name='Q13', avatar='🧑‍💻'):
        st.write(f"**13**.\tHave you used Cloud Computing Platforms on a regular basis? (Select all that apply): \t **{persona['cloud_compute']}**")
    with st.chat_message(name='Q14', avatar='🧑‍💻'):
        st.write(
            f"**14**.\tSelect any activities that make up an important part of your role at work: (Select all that apply)  \n **{persona['ml_activities']}**")
    with st.chat_message(name='Q15', avatar='🧑‍💻'):
        st.write(
            f"**15** . Approximately how many times have you used a TPU (tensor processing unit)?  \n **{persona['tpu_usage']}**")
    with st.chat_message(name='Q16', avatar='🧑‍💻'):
        st.write(
            f"**16** . Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?  \n **{persona['ml_spend']}**")
    with st.chat_message(name='Q17', avatar='🧑‍💻'):
        st.write(
            f"**17** . In what industry is your current employer/contract (or your most recent employer if retired)?  \n **{persona['industry']}**")
    with st.chat_message(name='Q18', avatar='🧑‍💻'):
        st.write(
            f"**18** . What is your current yearly compensation (approximate $USD)?  \n **{persona['income']}**")

    return

def handle_counts(ml_count:int, ml_framework:int,cv_algo:int, nlp_algo:int):
    i = random.randint(2, 5)
    ml_count = i
    ml_framework = i
    cv_algo = i
    nlp_algo = i
    return ml_count, ml_framework, cv_algo, nlp_algo

def handle_activities(activities:str):
    activities = [' Analyze and understand data to influence product or business decisions']
    return activities


def read_persona(persona:str) -> None:
    persona_dict = {}
    profile_df = pd.read_csv(data_dir.joinpath('Profiles.csv'),sep=',')
    test_df = pd.read_csv(data_dir.joinpath('Kaggle-Responses-Test.csv'))
    json_file = open(configs.joinpath('final-feats.json'))
    with open(configs.joinpath('col-mapping.yaml'), 'r') as stream:
        col_map = yaml.safe_load(stream)
    col_names = json.load(json_file)
    test_df.columns = col_names.values()
    profile_df.columns = ['Data Analyst', 'Data Engineer', 'Data Scientist', 'Machine Learning Engineer',
                          'Product/Project Manager', 'Research Scientist', 'Software Engineer', 'Statistician']
    data = profile_df[0:]
    persona_df = data[persona]
    top_index = np.argpartition(persona_df.values, -1)[-1:]
    print(top_index)
    top_entry =  test_df.iloc[top_index]
    education = [0,1,2,3,4]
    languages = [6,7,8,9,10,11]
    ide = [14, 15, 16, 17, 18, 19]
    activities_corp =[30, 31, 32, 33, 34, 35, 36, 37]
    industry = [41, 42, 43, 44, 45, 46, 47, 48]
    higher_education = boolean_single_select(top_entry.iloc[:, education])
    input_code_exp = int(top_entry.iloc[:,5].values)
    if input_code_exp ==0:
        input_code_exp = 4
    code_exp = col_map['code_experience'][input_code_exp]
    pgm_lang = boolean_multi_select(top_entry.iloc[:, languages],'Language')
    ide = boolean_multi_select(top_entry.iloc[:, ide], 'IDE')
    ml_input = int(top_entry.iloc[:,20].values)
    if ml_input == 0:
        ml_input = 3
    ml_exp = col_map['ml_experience'][ml_input]
    ml_framework_count = top_entry.iloc[:,21].values[0]
    ml_algo_count = top_entry.iloc[:, 22].values[0]
    cv_algo_count = top_entry.iloc[:, 23].values[0]
    nlp_algo_count = top_entry.iloc[:, 24].values[0]
    auto_ml_count = top_entry.iloc[:, 25].values[0]
    if persona == 'Data Scientist':
        ml_algo_count, ml_framework_count, cv_algo_count, nlp_algo_count = handle_counts(ml_algo_count, ml_framework_count, nlp_algo_count, cv_algo_count)

    business_intel_count = top_entry.iloc[:, 26].values[0]
    bd_count = top_entry.iloc[:, 27].values[0]
    managed_ml = top_entry.iloc[:, 28].values[0]
    cloud_compute = top_entry.iloc[:, 29].values[0]
    ml_activites = boolean_multi_select(top_entry.iloc[:, activities_corp], 'Activities')
    if persona == 'Data Analyst':
        ml_activites = handle_activities(ml_activites)
    tpu_usage = int(top_entry.iloc[:, 39].values)
    tpu_select =col_map['tpu_use'][tpu_usage]
    ml_spend = int(top_entry.iloc[:, 40].values)
    ml_spend_select = col_map['ml_spend'][ml_spend]
    industry_select = boolean_multi_select(top_entry.iloc[:, industry], "Industry")
    income = int(top_entry.iloc[:, 49].values)
    income_select = col_map['income'][income]
    persona_dict['education'] = higher_education
    persona_dict['coding'] = code_exp
    persona_dict['language'] = pgm_lang
    persona_dict['ide'] = ide
    persona_dict['ml_experience'] = ml_exp
    persona_dict['ml_framework'] = ml_framework_count
    persona_dict['ml_algo'] = ml_algo_count
    persona_dict['cv_count'] = cv_algo_count
    persona_dict['nlp_count'] = nlp_algo_count
    persona_dict['automl_count'] = auto_ml_count
    persona_dict['business_intel'] = business_intel_count
    persona_dict['big_data_products'] = bd_count
    persona_dict['managed_ml'] = managed_ml
    persona_dict['cloud_compute'] = cloud_compute
    persona_dict['ml_activities'] = ml_activites
    persona_dict['tpu_usage'] = tpu_usage
    persona_dict['ml_spend'] = ml_spend_select
    persona_dict['industry'] = industry_select
    persona_dict['income'] = income_select
    return persona_dict


def show_persona():
    st.subheader("Top Persona Outlook")
    personas = ['Data Analyst', 'Data Engineer', 'Data Scientist', 'Machine Learning Engineer',
                'Product/Project Manager', 'Research Scientist', 'Software Engineer', 'Statistician']
    sel_val = st.selectbox("Select Personas", personas)
    persona_dict = read_persona(sel_val)
    persona_card(persona_dict)

