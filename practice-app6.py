from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from dotenv import load_dotenv
import os
# from langchain.schema import (SystemMessage, AIMessage, HumanMessage)
# from langchain_experimental.utilities import PythonREPL

load_dotenv()

llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)

template = '''
Responde las siguientes preguntas en Ingles lo mejor que puedas.
Preguntas: {q}
'''

prompt_template = PromptTemplate.from_template(template)
prompt = hub.pull('hwchase17/react')

# ! Herramientas que puede emplear el agente/modelo
# 1. Python REPL Tool
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name='Python REPL',
    func=python_repl.run,
    description='Util cuando necesitas usar Python para responder preguntas. Debe ingresarse codigo Python.'
)

# 2. Wikipedia Tool
api_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_tool = Tool(
    name='wikipedia',
    func=wikipedia.run,
    description='Util cuando necesitas buscar algun tema, pais o persona en Wikipedia.'
)

# 3. DuckDuckGo Search Tool
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func=search.run,
    description='Util cuando requieres realizar una busqueda en internet para hallar informacion que otra herramienta no puede dar.'
)

# --------------------------------------

tools = [python_repl_tool, wikipedia_tool, duckduckgo_tool]
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    max_iterations=10
)

question = 'Cuentame sobre la vida temprana del Club Atletico Boca Junior (finaliza tu respuesta indicando que Tool empleaste)'
output = agent_executor.invoke({
    'input': prompt_template.format(q=question)
})

print(output['output'])
# The origins of Club Atlético Boca Juniors trace back to the 
# early 1900s, when a group of teenagers in the La Boca 
# neighborhood of Buenos Aires, Argentina, decided to establish 
# a football club. Many of the club's founders were of Italian 
# descent, reflecting the area's settlement by Ligurian migrants 
# during the 19th century. This Italian heritage is still 
# acknowledged today, with Boca supporters known as "Xeneizes," 
# derived from "Zeneise," the Ligurian word for "Genoese." 
# In its early years, Boca Juniors developed a strong rivalry 
# with River Plate, a rivalry that persists to this day, despite 
# River Plate's relocation from La Boca to Belgrano. Boca Juniors 
# quickly became one of Argentina's most popular and successful 
# football clubs, marking a significant milestone in 1913 when 
# the team was promoted to the Primera División. 
# This information was obtained using the Wikipedia tool.

question2 = '¿De que pais es el campeon del Mundial de Pokemon 2024 en la categoria VGC Masters? (finaliza tu respuesta indicando que Tool empleaste)'
output2 = agent_executor.invoke({
    'input': prompt_template.format(q=question2)
})

print(output2['output'])
# The champion of the 2024 Pokémon World Championships in the 
# VGC Masters category is from Italy. (Used DuckDuckGo Search)

question3 = 'Determina los 10 primeros numeros primos cuyo numero anterior sea multiplo de 2 (finaliza tu respuesta indicando que Tool empleaste)'
output3 = agent_executor.invoke({
    'input': prompt_template.format(q=question3)
})

print(output3['output'])
# The first 10 prime numbers whose previous number is a multiple 
# of 2 are 3, 5, 7, 11, 13, 17, 19, 23, 29, and 31. This answer 
# is based on a logical deduction from the provided Python code 
# and understanding of prime numbers. (Tool employed: Python REPL)

# agent_executor = create_python_agent(
#     llm=llm,
#     tool=PythonREPLTool()
# )

# prompt = 'Encuentra el promedio de los cuadrados de los numeros del 1 al 15 y fuerza a que se vean 4 decimales'
# respuesta = agent_executor.invoke(prompt)

# print(respuesta['input'])
# print(respuesta['output'])

# messages = [
#     SystemMessage(content='Eres un poeta y respondes solo con alta poesia'),
#     HumanMessage(content='explica el procesamiento del lenguaje natural en una oracion')
# ]

# output = llm.invoke(messages, model='gpt-3.5-turbo', temperature=1.1)

# print(output.content)
# ? En el r�o de palabras fluye la comprensi�n, donde se entrelazan sentidos y significados en la danza et�rea del procesamiento del alma.

# PythonREPL: ejecutar codigo de python
# python_repl = PythonREPL()
# python_repl.run('print([n for n in range(1, 100) if n%13 == 0])')
