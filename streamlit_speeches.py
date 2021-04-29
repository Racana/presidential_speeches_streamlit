import pandas as pd
import streamlit as st
from datetime import date 
import numpy as np

df = pd.read_csv('presidential_speeches.csv')
df.date = pd.to_datetime(df['date']).dt.date

df['speaker'] = np.where(df.date > date(2019,12,9), "Alberto Fernández", "Mauricio Macri")

st.sidebar.title("Discursos presidenciales Argentina")
st.sidebar.markdown(
    """
Transcritos de discursos presidenciales de Argentina desde Diciembre 2015, texto obtenido desde el [sitio de Casa Rosada ](https://www.casarosada.gob.ar/informacion/discursos)
para análisis utilizando novedosas técnicas de NLP (procesamiento natural del lenguaje).
Código utilizado en esta app pueden encontrarlos en [Github Pablo Racana](https://github.com/Racana/).
"""
)

speaker_list = df.speaker.unique().tolist()
speaker = st.sidebar.selectbox(
    'Orador', speaker_list
)

st.header("Discursos presidenciales")

speaker_frame = df.loc[df.speaker == speaker]

list_of_speeches =  [str(date)+" - "+title for date, title in zip(
    speaker_frame.date.values.tolist(), 
    speaker_frame.title.values.tolist()
    )
    ]

speech = st.selectbox(
    'Seleccione un discurso a continuación', list_of_speeches
)

st.subheader("Resumen")

summary = speaker_frame[speaker_frame.title == speech.split(" - ")[-1]]['summary'].values[0]
new_summary = summary.replace("\n", r"""   
""")

with st.beta_expander("Ver resumen"):
    st.markdown(new_summary)

with st.beta_expander("Ver metodología"):
    st.markdown("""
        El resumen anterior, fue obtenido seleccionando las oraciones principales del discurso, 
        mediante la vectorizacion de estas mediante el modelo BETO de la Universidad de Chile, 
        e identificando las principales oraciones calculando su similitud de coseno.  

        El primer paso consiste en limpiar el texto removiendo puntuaciones, símbolos y stopwords 
        (palabras vacías como los artículos, preposiciones, etc.) y luego separamos el discurso en oraciones.  

        Para el segundo paso tomamos cada oración y lo codificamos utilizando el modelo BETO, 
        desarrollado por la Universidad de Chile basado en el modelo BERT de Google.
        BETO, BERT o cualquier encoder, básicamente transforma cada oración en un vector de números, 
        tomando en consideración la oración completa, y no palabra por palabra.  
        """)

    st.image("https://nextjournal.com/data/QmaXGoXqdcjopjVzG1xAQQMS1BsEECSEoxRgLqoeDdetv7?content-type=image%2Fpng&filename=2019-06-12%2000-50-16%20%E7%9A%84%E8%9E%A2%E5%B9%95%E6%93%B7%E5%9C%96.png")

    st.markdown("""
        Paso siguiente, tomamos cada vector generado por el encoder y la distancia de coseno con el resto de los vectores (oraciones), 
        lo que nos dará una matriz de NxN, donde N es la cantidad de oraciones que estamos analizando, 
        y cada celda tendrá el valor de similitud de la oración M_1 comparada con la oración M_2.
        """)
    
    st.image("https://datascience-enthusiast.com/figures/cosine_sim.png")
    
    st.markdown("""
        Finalmente, para cada oración, calculamos el promedio de similitud con el resto de las oraciones, 
        los que nos da un vector de 1xN, luego, seleccionamos aquellas oraciones con mayor similitud las cuales son 
        las que concentran la idea principal del texto y le agregamos el título como primera oración para mayor contexto. 
        Las oraciones seleccionadas son las que se muestran en el recuadro anterior.
        """)

    


st.subheader("Discurso")

body = speaker_frame[speaker_frame.title == speech.split(" - ")[-1]]['body'].values[0]
new_body = body.replace("\n", r"""   
""")
st.markdown(new_body)