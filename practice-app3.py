from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

template = "Eres un asistente que traduce del {idioma_entrada} al {idioma_salida} el escrito: {texto}."

texto = "Me encanta programar."

prompt_template = PromptTemplate(
    input_variables=["idioma_entrada", "idioma_salida", "texto"],
    template=template
)

# Crear modelo de lenguaje | temperature -> 0 (deterministico) / temperature -> 2 (creativo)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Crear Cadena
chain = LLMChain(llm=llm, prompt=prompt_template)

response = chain.invoke(input={"idioma_entrada":"español", "idioma_salida":"francés", "texto":texto})
print(response)
