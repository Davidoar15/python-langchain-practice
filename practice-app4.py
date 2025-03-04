from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# import requests
import os

load_dotenv()

# ? LangChain Version

def movies(movie):
    import requests

    url = f"https://api.themoviedb.org/3/search/movie?query={movie}&include_adult=false&language=en-US&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2MTU4OTYwNTZjNDgyODI3NzQxOWUzODZhM2IxY2IzNSIsIm5iZiI6MTc0MDE0ODAyMC43MTEsInN1YiI6IjY3Yjg4ZDM0YTIyODQ2NjZmMWViNmEwMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.1TmPKRBWm0LK30J1iMNDqSQQoT-ZUEuFKwtK1GGy5Tk"
    }

    response = requests.get(url, headers=headers)   
    return response

template = """Te compartire informacion sobre algunas peliculas. Debes aportar informacion (en italiano)
del titulo, fecha de estreno y resumen de las primeras 3 que aparezcan de forma estructurada (si aparecen menos, aporta las que aparezcan).
{response}"""

prompt_template = PromptTemplate(
    input_variables=["response"], 
    template=template
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# ! LLMChain con RunnableSequence
chain = prompt_template | llm

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chain = LLMChain(llm=llm, prompt=prompt_template)

response = movies("sonic")

# ! invoke() en lugar de run()
print(chain.invoke({"response": response.text}))

# print(chain.run(response=response.text))

# ? Classic Version

# peli = "sonic"
# url = f"https://api.themoviedb.org/3/search/movie?query={peli}&include_adult=false&language=en-US&page=1"

# headers = {
#     "accept": "application/json",
#     "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2MTU4OTYwNTZjNDgyODI3NzQxOWUzODZhM2IxY2IzNSIsIm5iZiI6MTc0MDE0ODAyMC43MTEsInN1YiI6IjY3Yjg4ZDM0YTIyODQ2NjZmMWViNmEwMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.1TmPKRBWm0LK30J1iMNDqSQQoT-ZUEuFKwtK1GGy5Tk"
# }

# response = requests.get(url, headers=headers)   
# results = response.json()

# for i in range(min(len(results["results"]), 3)):
#     title = results["results"][i]["original_title"]
#     overview = results["results"][i]["overview"]
#     date = results["results"][i]["release_date"]
#     ! """ -> Mejor legibilidad sin usar \n
#     print(f"""Titulo: {title}
#     Fecha de estreno: {date}
#     Resumen: {overview}
#     """)
