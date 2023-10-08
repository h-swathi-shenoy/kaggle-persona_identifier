import streamlit as st
from omegaconf import OmegaConf
from src.pathfinder import PathConfig
import requests
import numpy as np
import json

path_obj = PathConfig()
config_dir = path_obj.configs_dir
col_map = OmegaConf.load(config_dir.joinpath('col-mapping.yaml'))


def count_inputs(option: list) -> int:
    """

    Parameters
    :param option:List
    ----------
    option

    Returns:
    count of choices selected for a multichoice question.
    -------

    """
    attempted_qs = [q for q in option if q == True]
    return len(attempted_qs)


def on_checkbox_select(chk_input: bool) -> int:
    """
    Marks 1 if the question is answered else 0.

    Parameters
    ----------
    chk_input : bool

    Returns : int
    -------
   """
    if chk_input:
        return 1
    else:
        return 0


def question_answer_check(checks: list) -> int:
    """
    Check if a particular question is attempted or no. If attempted then increment the variable, else unattempted.
    Parameters
    ----------
    checks : list of multiple choice values for a question.

    Returns
    -------
    int
    """
    answered = 0
    for i in checks:
        if i == 1 or i == True:
            answered = +1
            pass
    return answered


def all_answered_chk(answer: list) -> bool:
    """
    Final attempted check on list of questions before clicking the submit button.

    Parameters
    ----------
    answer : List of all questions with values 1 or 0.

    Returns
    -------
    all_answered : int
    1 if all questions attempted else 0.

    """
    all_answered = 0
    for i in answer:
        if i != 1:
            return all_answered
        else:
            all_answered = 1
    return all_answered


def api_input_design(api_inputs: list) -> json:
    """
    Prepare the inputs to the questionnaire to the Post method for the API.
    Parameters
    ----------
    api_inputs : list

    Returns: json

    Json Inputs to the API.
    -------

    """
    float_list = [float(i) for i in api_inputs]
    input_arr = np.array(float_list)
    json_ip = dict(enumerate(input_arr.flatten(), 1))
    return json_ip


def run_questionnaire():
    """
    Questionnaire for Look at top persona tab, validates the inputs and send them to the API gateway for persona prediction.
    """
    st.subheader("Questionnaire")
    st.write(
        f"**1**. What is the highest level of formal education that you have attained/or plan to attain in next 2 years?")
    q1_1 = st.checkbox('High School', key='q1_1')
    q1_2 = st.checkbox('Bachelors', key='q1_2')
    q1_3 = st.checkbox('Masters', key='q1_3')
    q1_4 = st.checkbox('Professional Degree', key='q1_4')
    q1_5 = st.checkbox('Doctorate', key='q1_5')
    q1_1_val = on_checkbox_select(q1_1)
    q1_2_val = on_checkbox_select(q1_2)
    q1_3_val = on_checkbox_select(q1_3)
    q1_4_val = on_checkbox_select(q1_4)
    q1_5_val = on_checkbox_select(q1_5)
    code_exp = col_map['code_experience']
    q2 = st.selectbox(
        f'**2**.  For how many years have you been writing code and/or programming?',
        code_exp.keys(), format_func=lambda x: code_exp[x])

    st.write(f"**3**. What programming languages do you use on a regular basis? (Select all that apply)")
    q3_1 = st.checkbox("Python", key="q3_1")
    q3_2 = st.checkbox("R", key="q3_2")
    q3_3 = st.checkbox("SQL", key="q3_3")
    q3_4 = st.checkbox("C", key="q3_4")
    q3_5 = st.checkbox("C++", key="q3_5")
    q3_6 = st.checkbox("Javascript", key="q3_6")
    q3_7 = st.checkbox("MATLAB", key="q3_7")
    q3_8 = st.checkbox("None", key="q3_8")
    q3_1_val = on_checkbox_select(q3_1)
    q3_2_val = on_checkbox_select(q3_2)
    q3_3_val = on_checkbox_select(q3_3)
    q3_4_val = on_checkbox_select(q3_4)
    q3_5_val = on_checkbox_select(q3_5)
    q3_6_val = on_checkbox_select(q3_6)
    q3_7_val = on_checkbox_select(q3_7)
    q3_8_val = on_checkbox_select(q3_8)

    st.write(
        "**4**. Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply)")
    q4_1 = st.checkbox("JupyterLab, Jupyter Notebooks", key="q4_1")
    q4_2 = st.checkbox("RStudio", key="q4_2")
    q4_3 = st.checkbox("Visual Studio", key="q4_3")
    q4_4 = st.checkbox("PyCharm", key="q4_4")
    q4_5 = st.checkbox("MATLAB", key="q4_5")
    q4_6 = st.checkbox("None", key="q4_6")
    q4_1_val = on_checkbox_select(q4_1)
    q4_2_val = on_checkbox_select(q4_2)
    q4_3_val = on_checkbox_select(q4_3)
    q4_4_val = on_checkbox_select(q4_4)
    q4_5_val = on_checkbox_select(q4_5)
    q4_6_val = on_checkbox_select(q4_6)

    st.write(
        "**5**.\tWhich of the following Machine Learning Frameworks do you use on a regular basis? (Select all that apply)")
    q5_1 = st.checkbox("Scikit-Learn", key='q5_1')
    q5_2 = st.checkbox("Lightgbm", key='q5_2')
    q5_3 = st.checkbox("Xgboost", key='q5_3')
    q5_4 = st.checkbox("Caret", key='q5_4')
    q5_5 = st.checkbox("Tidymodels", key='q5_5')
    q5_6 = st.checkbox("Prophet", key='q5_6')
    q5_7 = st.checkbox("Tensorflow", key='q5_7')
    q5_8 = st.checkbox("Keras", key='q5_8')
    q5_9 = st.checkbox("Pytorch", key='q5_9')
    q5_10 = st.checkbox("Fast.api", key='q5_10')
    q5_11 = st.checkbox("None", key='q5_11')

    st.write(
        "**6**.\tWhich of the following Machine Learning Algorithms do you use on a regular basis? (Select all that apply)")
    q6_1 = st.checkbox("Linear/Logistic", key='q6_1')
    q6_2 = st.checkbox("Tree Based(DT/Random Forest)", key='q6_2')
    q6_3 = st.checkbox("Gradient Boost Machines", key='q6_3')
    q6_4 = st.checkbox("Bayesian Approaches", key='q6_4')
    q6_5 = st.checkbox("Evolution Approaches", key='q6_5')
    q6_6 = st.checkbox("Dense Neural Networks", key='q6_6')
    q6_7 = st.checkbox("Convolution Networks", key='q6_7')
    q6_8 = st.checkbox("GAN's", key='q6_8')
    q6_9 = st.checkbox("RNN's", key='q6_9')
    q6_10 = st.checkbox("Transformer Networks", key='q6_10')
    q6_11 = st.checkbox("None", key='q6_11')

    st.write(
        "**7**.\tWhich of the following Computer Vision Algorithms do you use on a regular basis? (Select all that apply)")
    q8_1 = st.checkbox("General Purpose image/video tools", key='q8_1')
    q8_2 = st.checkbox("Image Segmentation Methods", key='q8_2')
    q8_3 = st.checkbox("Object Detection Methods", key='q8_3')
    q8_4 = st.checkbox("Image Classifaction", key='q8_4')
    q8_5 = st.checkbox("GAN", key='q8_5')
    q8_6 = st.checkbox("None", key='q8_6')

    st.write(
        "**8**.\tWhich of the following NLP Algorithms do you use on a regular basis? (Select all that apply)")
    q9_1 = st.checkbox("Word Embedding/Vectors", key='q9_1')
    q9_2 = st.checkbox("Encoder-Decoder", key='q9_2')
    q9_3 = st.checkbox("Contextualized Embeddings", key='q9_3')
    q9_4 = st.checkbox("Transformer Language Models", key='q9_4')
    q9_5 = st.checkbox("GAN", key='q9_5')
    q9_6 = st.checkbox("None", key='q9_6')

    st.write(
        "**9**.\tWhich of the following Auto ML tools do you use on a regular basis? (Select all that apply)")
    q10_1 = st.checkbox("Automated Data Augmentation", key='q10_1')
    q10_2 = st.checkbox("Automated Feature Eng/Selection", key='q10_2')
    q10_3 = st.checkbox("Automated Model Selection", key='q10_3')
    q10_4 = st.checkbox("Automated Model Architecture", key='q10_4')
    q10_5 = st.checkbox("Automated Hyperparameter Tuning", key='q10_5')
    q10_6 = st.checkbox("Automated Full Model Pipeline", key='q10_6')
    q10_7 = st.checkbox("None", key = 'q10_7')
    st.write(
        "**10**.\tWhich of the following Business Intelligence tools do you use on a regular basis? (Select all that apply)")
    q11_1 = st.checkbox("Power BI", key='q11_1')
    q11_2 = st.checkbox("Amazon Quicksight", key='q11_2')
    q11_3 = st.checkbox("Google DataStudio", key='q11_3')
    q11_4 = st.checkbox("Looker", key='q11_4')
    q11_5 = st.checkbox("Tableau", key='q11_5')
    q11_6 = st.checkbox("Salesforce", key='q11_6')
    q11_7 = st.checkbox("Domo", key='q11_7')
    q11_8 = st.checkbox("TIBCO Software", key='q11_8')
    q11_9 = st.checkbox("Sisense", key='q11_9')
    q11_10 = st.checkbox("SAP", key='q11_10')
    q11_11 = st.checkbox("None", key='q11_11')
    st.write(
        "**11**.\tWhich of the following Business Data Products do you use on a regular basis? (Select all that apply)")
    q13_1 = st.checkbox("MQSQL", key='q13_1')
    q13_2 = st.checkbox("PostgreSQL", key='q13_2')
    q13_3 = st.checkbox("SQLite", key='q13_3')
    q13_4 = st.checkbox("Oracle DB", key='q13_4')
    q13_5 = st.checkbox("MongoDB", key='q13_5')
    q13_6 = st.checkbox("Snowflake", key='q13_6')
    q13_7 = st.checkbox("IBM DB2", key='q13_7')
    q13_8 = st.checkbox("Microsoft SQL Server", key='q13_8')
    q13_9 = st.checkbox("Amazon Redshift/DynamoDB", key='q13_9')
    q13_10 = st.checkbox("Google Cloud", key='q13_10')
    q13_11 = st.checkbox("None", key='q13_11')
    st.write(
        "**12**.\tDo you use any following managed ML products on a regular basis? (Select all that apply)")
    q14_1 = st.checkbox("Amazon Sagemaker", key='q14_1')
    q14_2 = st.checkbox("Azure Machine Learning Studio", key='q14_2')
    q14_3 = st.checkbox("None", key='q14_3')
    st.write(
        "**13**.\tWhich of the following Cloud Computing Platforms do you use on a regular basis? (Select all that apply)")
    q15_1 = st.checkbox("AWS", key='q15_1')
    q15_2 = st.checkbox("Microsoft Azure", key='q15_2')
    q15_3 = st.checkbox("Google Cloud Platform", key='q15_3')
    q15_4 = st.checkbox("IBM Cloud", key='q15_4')
    q15_5 = st.checkbox("Oracle Cloud", key='q15_5')
    q15_6 = st.checkbox("Snowflake", key='q15_6')
    q15_7 = st.checkbox("IBM DB2", key='q15_7')
    q15_8 = st.checkbox("Microsoft SQL Server", key='q15_8')
    q15_9 = st.checkbox("Amazon Redshift/DynamoDB", key='q15_9')
    q15_10 = st.checkbox("Google Cloud", key='q15_10')
    q15_11 = st.checkbox("None", key='q15_11')
    st.write(
        "**14**.\tSelect any activities that make up an important part of your role at work: (Select all that apply)")
    q16_1 = st.checkbox("Analyze and understand data to influence product or business decisions", key='q16_1')
    q16_2 = st.checkbox(
        "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data",
        key='q16_2')
    q16_3 = st.checkbox("Build prototypes to explore applying machine learning to new areas", key='q16_3')
    q16_4 = st.checkbox(
        "Build and/or run a machine learning service that operationally improves my product or workflow", key='q16_4')
    q16_5 = st.checkbox("Experimentation and iteration to improve existing ML models", key='q16_5')
    q16_6 = st.checkbox("Do research that advances the state of the art of machine learning", key='q16_6')
    q16_7 = st.checkbox("None of these activities are an important part of my role at work", key='q16_7')
    q16_8 = st.checkbox("Other", key='q16_8')
    q16_1_val = on_checkbox_select(q16_1)
    q16_2_val = on_checkbox_select(q16_2)
    q16_3_val = on_checkbox_select(q16_3)
    q16_4_val = on_checkbox_select(q16_4)
    q16_5_val = on_checkbox_select(q16_5)
    q16_6_val = on_checkbox_select(q16_6)
    q16_7_val = on_checkbox_select(q16_7)
    q16_8_val = on_checkbox_select(q16_8)

    ml_exp = col_map['ml_experience']
    q17 = st.selectbox(
        f'**15**.\tFor how many years have you used machine learning methods?', ml_exp.keys(),
        format_func=lambda x: ml_exp[x])

    tpu_use = col_map['tpu_use']
    q18 = st.selectbox(
        f'**16**.\tApproximately how many times have you used a TPU (tensor processing unit)?', tpu_use.keys(),
        format_func=lambda x: tpu_use[x])

    ml_spend = col_map['ml_spend']
    q19 = st.selectbox(
        f'**17**.\tApproximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?',
        ml_spend.keys(),
        format_func=lambda x: ml_spend[x])

    st.write(
        "**18**.\tIn what industry is your current employer/contract (or your most recent employer if retired)?")
    q20_1 = st.checkbox("Academics/Education", key='q20_1')
    q20_2 = st.checkbox("Computers/Technology", key='q20_2')
    q20_3 = st.checkbox("Insurance/Risk Assessment", key='q20_3')
    q20_4 = st.checkbox("Manufacturing/Fabrication", key='q20_4')
    q20_5 = st.checkbox("Medical/Pharmaceutica", key='q20_5')
    q20_6 = st.checkbox("Online Service/Internet-based Service", key='q20_6')
    q20_7 = st.checkbox("Retail/Sales", key='q20_7')
    q20_8 = st.checkbox("Other", key='q20_8')
    q20_1_val = on_checkbox_select(q20_1)
    q20_2_val = on_checkbox_select(q20_2)
    q20_3_val = on_checkbox_select(q20_3)
    q20_4_val = on_checkbox_select(q20_4)
    q20_5_val = on_checkbox_select(q20_5)
    q20_6_val = on_checkbox_select(q20_6)
    q20_7_val = on_checkbox_select(q20_7)
    q20_8_val = on_checkbox_select(q20_8)

    income = col_map['income']
    q21 = st.selectbox(
        f'**19**.\tWhat is your current yearly compensation (approximate $USD)',
        income.keys(),
        format_func=lambda x: income[x])

    q5_count = count_inputs([q5_1, q5_2, q5_3, q5_4, q5_5, q5_6, q5_7, q5_8, q5_9, q5_10, q5_11])
    q6_count = count_inputs([q6_1, q6_2, q6_3, q6_4, q6_5, q6_6, q6_7, q6_8, q6_9, q6_10, q6_11])
    q8_count = count_inputs([q8_1, q8_2, q8_3, q8_4, q8_5])
    q9_count = count_inputs([q9_1, q9_2, q9_3, q9_4, q9_5])
    q10_count = count_inputs([q10_1, q10_2, q10_3, q10_4, q10_5, q10_6])
    q11_count = count_inputs([q11_1, q11_2, q11_3, q11_4, q11_5, q11_6, q11_7, q11_8, q11_9, q11_10])
    q13_count = count_inputs([q13_1, q13_2, q13_3, q13_4, q13_5, q13_6, q13_7, q13_8, q13_9, q13_10])
    q14_count = count_inputs([q14_1, q14_2])
    q15_count = count_inputs([q15_1, q15_2, q15_3, q15_4, q15_5, q15_6, q15_7, q15_8, q15_9, q15_10])

    if st.button("Submit", type='primary'):
        q1_ans = question_answer_check([q1_1_val,q1_2_val, q1_3_val, q1_4_val, q1_5_val])
        q3_ans = question_answer_check([q3_1_val, q3_2_val, q3_3_val,q3_4_val, q3_5_val, q3_6_val, q3_7_val, q3_8_val])
        q4_ans = question_answer_check([q4_1_val, q4_2_val, q4_3_val, q4_4_val, q4_5_val, q4_6_val])
        q5_ans = question_answer_check([q5_1, q5_2, q5_3, q5_4, q5_5, q5_6, q5_7, q5_8, q5_9, q5_10, q5_11])
        q6_ans = question_answer_check([q6_1, q6_2, q6_3, q6_4, q6_5, q6_6, q6_7, q6_8, q6_9, q6_10, q6_11])
        q8_ans = question_answer_check([q8_1, q8_2, q8_3, q8_4, q8_5, q8_6])
        q9_ans = question_answer_check([q9_1, q9_2, q9_3, q9_4, q9_5,q9_6])
        q10_ans = question_answer_check([q10_1, q10_2, q10_3, q10_4, q10_5, q10_6,q10_7])
        q11_ans = question_answer_check([q11_1, q11_2, q11_3, q11_4, q11_5, q11_6, q11_7, q11_8, q11_9, q11_10, q11_11])
        q13_ans = question_answer_check([q13_1, q13_2, q13_3, q13_4, q13_5, q13_6, q13_7, q13_8, q13_9, q13_10, q13_11])
        q14_ans = question_answer_check([q14_1, q14_2, q14_3])
        q15_ans = question_answer_check([q15_1, q15_2, q15_3, q15_4, q15_5, q15_6, q15_7, q15_8, q15_9, q15_10, q15_11])
        all_answered:bool = all_answered_chk([q1_ans, q3_ans, q4_ans, q5_ans, q6_ans, q8_ans,
                                             q9_ans, q10_ans, q11_ans, q13_ans, q14_ans, q15_ans])
        if all_answered:
            flatten_arr = api_input_design([q1_1_val, q1_2_val, q1_3_val, q1_4_val, q1_5_val,
                                            q2, q3_1_val, q3_2_val, q3_3_val, q3_4_val, q3_5_val, q3_6_val, q3_7_val,
                                            q3_8_val,
                                            q4_1_val, q4_2_val, q4_3_val, q4_4_val, q4_5_val, q4_6_val, q17, q5_count,
                                            q6_count,
                                            q8_count, q9_count, q10_count, q11_count, q13_count, q14_count, q15_count,
                                            q16_1_val, q16_2_val, q16_3_val, q16_4_val, q16_5_val, q16_6_val, q16_7_val,
                                            q16_8_val,q17,
                                            q18, q19, q20_1_val, q20_2_val, q20_3_val, q20_4_val, q20_5_val, q20_6_val,
                                            q20_7_val, q20_8_val,
                                            q21
                                            ])
            uri = "https://9d9767xtv8.execute-api.us-east-1.amazonaws.com/dev"
            headers = {'Content-type': 'application/json'}
            persona_response = requests.post(uri, data=json.dumps(flatten_arr), headers=headers)
            persona_response = json.loads(persona_response.json()['result'])
            persona = list(persona_response.values())[0]
            probability = list(persona_response.values())[1]
            st.text_area(label="**Persona Prediction**", value=persona, height=25, disabled=True, key='resp1')
            st.text_area(label="**Persona Probability(%)**", value=int(np.round(probability, 0)), height=25, disabled=True,
                         key='resp2')
        else:
            st.toast("Please attempt all the questions.")

if __name__ == '__main__':
    questionnaire()
