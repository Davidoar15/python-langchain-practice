from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SimpleSequentialChain
from dotenv import load_dotenv
import os

load_dotenv()

# ? Sequential Chain

llm_1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
template_1 = "Eres un ingeniero en sistemas muy experimentado y tambien un excelente programador en TypeScript. Estas dando una clase libre en una universidad. Escribe de forma optima una funcion que implemente el concepto de {concepto}"
prompt_template_1 = PromptTemplate.from_template(
    template=template_1
)
chain_1 = prompt_template_1 | llm_1

llm_2 = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=1.2)
template_2 = "Dada la funcion de TypeScript {funcion}, describela para tus alumnos de universidad lo mas detalladamente posible."
prompt_template_2 = PromptTemplate.from_template(
    template=template_2
)
chain_2 = prompt_template_2 | llm_2

# ! No funcional con RunneableSequence
# chain = SimpleSequentialChain(chains=[chain_1, chain_2])

chain = chain_1 | chain_2

output = chain.invoke('Recursividad')

print(output.content)

# ? Simple Chain

# llm = ChatOpenAI()

# template = """Eres un entrenador pokemon. Describe tu estrategia para derrotar un rival que posee un
# {pkmn_rival} y tu pokemon es {pkmn}."""

# prompt_template = PromptTemplate.from_template(template=template)

# chain = prompt_template | llm

# output = chain.invoke({'pkmn_rival': 'Garchomp', 'pkmn': 'Weavile'})

# print(output.content)