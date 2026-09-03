import streamlit as st
import pandas as pd
import pickle


# =========================
# LOAD MODEL
# =========================
with open('model_svm.pkl', 'rb') as file:
    model_svm = pickle.load(file)


# =========================
# MAIN FUNCTION
# =========================
def run():

    st.title('Visa Application Status Prediction')

    st.write(
        'Fill in the information below to predict whether the visa application '
        'is likely to be Certified or Denied.'
    )


    # =========================
    # SESSION STATE
    # =========================
    if 'input_history' not in st.session_state:
        st.session_state.input_history = pd.DataFrame()


    # =========================
    # INPUT FORM
    # =========================
    with st.form('visa_form'):

        case_id = st.text_input(
            'Case ID',
            value='EZYV300000'
        )

        continent = st.selectbox(
            'Continent',
            ('Asia',)
        )

        education = st.selectbox(
            'Education of Employee',
            (
                'High School',
                "Bachelor's",
                "Master's",
                'Doctorate'
            )
        )

        job_experience = st.selectbox(
            'Has Job Experience?',
            ('Y', 'N')
        )

        job_training = st.selectbox(
            'Requires Job Training?',
            ('Y', 'N')
        )

        no_of_employees = st.number_input(
            'Number of Employees',
            min_value=1,
            value=2500
        )

        yr_of_estab = st.number_input(
            'Year of Establishment',
            min_value=1800,
            max_value=2026,
            value=2000
        )

        region = st.selectbox(
            'Region of Employment',
            (
                'Northeast',
                'West',
                'South',
                'Midwest',
                'Island'
            )
        )

        prevailing_wage = st.number_input(
            'Prevailing Wage',
            min_value=0.0,
            value=85000.0
        )

        unit_of_wage = st.selectbox(
            'Unit of Wage',
            ('Year',)
        )

        full_time = st.selectbox(
            'Full Time Position?',
            ('Y', 'N')
        )

        submitted = st.form_submit_button('Predict')


    # =========================
    # CREATE INPUT DATA
    # =========================
    data_inf = {
        'case_id': case_id,
        'continent': continent,
        'education_of_employee': education,
        'has_job_experience': job_experience,
        'requires_job_training': job_training,
        'no_of_employees': no_of_employees,
        'yr_of_estab': yr_of_estab,
        'region_of_employment': region,
        'prevailing_wage': prevailing_wage,
        'unit_of_wage': unit_of_wage,
        'full_time_position': full_time
    }

    data_inf = pd.DataFrame([data_inf])


    # =========================
    # PREDICTION
    # =========================
    if submitted:

        prediction = model_svm.predict(data_inf)

        result = data_inf.copy()
        result['case_status'] = prediction


        # =========================
        # ADD / UPDATE DATA
        # =========================
        st.session_state.input_history = pd.concat(
            [
                st.session_state.input_history,
                result
            ],
            ignore_index=True
        )

        # Kalau case_id sama,
        # simpan data yang paling baru
        st.session_state.input_history = (
            st.session_state.input_history
            .drop_duplicates(
                subset='case_id',
                keep='last'
            )
            .reset_index(drop=True)
        )


        # =========================
        # PREDICTION RESULT
        # =========================
        st.write('### Prediction Result')

        if prediction[0] == 'Certified':
            st.success('Certified')

        else:
            st.error('Denied')


    # =========================
    # INPUT DATA HISTORY
    # =========================
    st.write('### Input Data')

    if len(st.session_state.input_history) > 0:

        st.dataframe(
            st.session_state.input_history,
            use_container_width=True
        )

    else:

        st.info('No prediction data yet.')


# =========================
# RUN
# =========================
if __name__ == '__main__':
    run()