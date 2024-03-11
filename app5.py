from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st  
from langchain.text_splitter import CharacterTextSplitter  
from langchain.embeddings.openai import OpenAIEmbeddings  
#Cambiamos el FAISS por Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain  
from langchain.llms import OpenAI  
from langchain.callbacks import get_openai_callback  
import langchain

langchain.verbose = False  

load_dotenv()

def process_text(text):
  
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
# Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# En lugar de usar FAISS.from_text lo cambiamos por Chroma.form_texts
  knowledge_base = Chroma.from_texts(chunks, embeddings)
  return knowledge_base
  
  
# Función principal de la aplicación
def main():
  
  #Le añadimos título, descripciones e icónos para que la interfaz gráfica sea mas agradable al usuario
  st.title("BIENVENIDO..!!! :wave:")
  st.markdown("Esta aplicación te ayudará a clasificar textos")
  st.title("Clasificación de un PDF :📑:")  
  texto= st.text_input #Añadimos el texto de entrada
  pdf = st.file_uploader("Sube tu archivo PDF a clasificar", type="pdf")  # Crea un cargador de archivos para subir archivos PDF
  st.write("Para clasificar tu texto, ingresa las categorías que deseas consultar en el siguiente espacio, una vez hallas cargado tu archivo")
  
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    # Almacena el texto del PDF en una variable
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    knowledge_base = process_text(text)
    query = st.text_input('Escribe las categorias de clasificación para el PDF...')
    cancel_button = st.button('Cancelar')

    if cancel_button:
      st.stop() 

    if query:
      docs = knowledge_base.similarity_search(query)
      # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
      model = "gpt-3.5-turbo-instruct" # Acepta 4096 tokens
      temperature = 0 
      llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
      chain = load_qa_chain(llm, chain_type="stuff")

      with get_openai_callback() as cost:
        response = chain.invoke(input={"question": query, "input_documents": docs})
        print(cost)  # Imprime el costo de la operación
        st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación
        st.markdown("**** TOKENS CONSUMIDOS Y VALOR TOTAL***")
        st.markdown("✅")
        st.markdown(cost)
# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
  main()  # Llama a la función principal
