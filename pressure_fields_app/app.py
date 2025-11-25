# app.py
import streamlit as st


st.set_page_config(page_title="UJO demo", layout="wide")
st.title("UJO demo")
pg = st.navigation([st.Page("rotordynamics_page.py", title = "rotordynamics page"),
                   st.Page("pressure_fields_page.py", title="pressure fields page")])

pg.run()
