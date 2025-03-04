from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# una Plantilla
prompt = PromptTemplate.from_template("Describe un objeto que te resulte {adjetivo} y por que tiene ese efecto en ti.")

#prompt.format(adjetivo="divertido") | Cambiar variable
print(prompt.format(adjetivo="divertido"))
