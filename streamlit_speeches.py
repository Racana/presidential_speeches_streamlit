import pandas as pd
import streamlit as st
from datetime import date 
import numpy as np
import torch

@st.cache()
def load_paragraphs():
    """
    load csv files using Streamlit cache function
    """
    df = pd.read_csv(r'data/presidential_speeches_paragraphs.csv', index_col=0)

    return df

@st.cache
def load_sentences():
    """
    load csv files using Streamlit cache function
    """
    df = pd.read_csv(r'data/presidential_speeches_sentences.csv', index_col=0)

    return df

@st.cache
def load_promises():
    """
    load csv files using Streamlit cache function
    """
    df = pd.read_csv(r'data/promises.csv')

    return df

@st.cache
def load_tensor(filename):
    """
    load tensor files using Streamlit cache function
    """
    tensor = torch.load(filename)

    return tensor

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    return sim_mt

paragraphs_df = load_paragraphs()
sentences_df = load_sentences()
promises_df = load_promises()

promises_embeddings = load_tensor(r'data/promises_embeddings.pt')
sentences_embeddings = load_tensor(r'data/sentences_embeddings.pt')

st.sidebar.title("Discursos presidenciales Argentina")
st.sidebar.markdown(
    """
Transcritos de discursos presidenciales de Argentina desde Diciembre 2015, texto obtenido desde el [sitio de Casa Rosada ](https://www.casarosada.gob.ar/informacion/discursos)
para análisis utilizando novedosas técnicas de NLP (procesamiento natural del lenguaje).
Código utilizado en esta app pueden encontrarlos en [Github Pablo Racana](https://github.com/Racana/).
"""
)

speaker_list = promises_df.speaker.unique()
speaker = st.sidebar.selectbox(
    'Orador', speaker_list
)

st.header("Promesa presidencial")

list_of_promises =  promises_df.loc[promises_df['speaker'] == speaker, 'short_description'].values.tolist()
speech = st.selectbox(
    'Seleccione una promesa a continuación', list_of_promises
)

promise_index = promises_df.short_description.values.tolist().index(speech)

promise = promises_df.loc[promise_index, 'promise']
st.markdown("Promesa del presidente " + speaker + ' "' + promise + '"')

st.subheader("Menciones en discursos")

sim_promise = sim_matrix(promises_embeddings, sentences_embeddings)
top_value, top_index = torch.topk(sim_promise[promise_index], 1)

print(top_value, top_index)

uuid_top_result = sentences_df.index.values[top_index.item()]
paragraph = paragraphs_df.loc[uuid_top_result, 'paragraphs']
event = paragraphs_df.loc[uuid_top_result, 'title']

paragraph = sentences_df.sentences[top_index.item()]

with st.beta_expander(event):
    st.markdown(paragraph)

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

