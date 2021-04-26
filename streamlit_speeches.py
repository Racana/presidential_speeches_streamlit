import pandas as pd
import streamlit as st
from datetime import date 
import numpy as np

df = pd.read_csv('presidential_speeches.csv')
df.date = pd.to_datetime(df['date']).dt.date

df['speaker'] = np.where(df.date > date(2019,12,9), "Alberto Fernández", "Mauricio Macri")


st.sidebar.title("Argentinian Presidential Speeches")
st.sidebar.markdown(
    """
Argentinian Presidential Speeches text obtained from [Casa Rosada site](https://www.casarosada.gob.ar/informacion/discursos)
for analysis using novel NLP techniques.  
Code can be viewed in [Pablo Racana's Github](https://github.com/Racana/).
"""
)

speaker_list = df.speaker.unique().tolist()
speaker = st.sidebar.selectbox(
    'Speaker', speaker_list
)

st.header("Presidential Speech")
st.subheader("Select one of the speaches below")

speaker_frame = df.loc[df.speaker == speaker]

list_of_speeches =  [str(date)+" - "+title for date, title in zip(
    speaker_frame.date.values.tolist(), 
    speaker_frame.title.values.tolist()
    )
    ]

speech = st.selectbox(
    'Select Presidential Speech', list_of_speeches
)

body = speaker_frame[speaker_frame.title == speech.split(" - ")[-1]]['body'].values[0]
new_body = body.replace("\n", r"""   
""")
st.markdown(new_body)