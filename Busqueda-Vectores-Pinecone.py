from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import os
import random
import pinecone

load_dotenv(find_dotenv(), override=True)

pc = pinecone.Pinecone()
index_name = 'langchain-practice'
# print(pc.describe_index(index_name))

# ! Splitting and Embedding Texts (Similarity Search)
with open('examples-txt/1984.txt', encoding='utf-8') as f:
    text = f.read()

# Fragmentacion del texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len
)

chunks = text_splitter.create_documents([text])
# print(chunks[7])
# print(chunks[7].page_content)
# print(f'Hay {len(chunks)} fragmentos')

# ! Crear Embedding

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)

# ? Un Chunk
# vector = embeddings.embed_query(chunks[0].page_content)

# indexes = pc.list_indexes().names()
# for i in indexes:
#     print('Borrando ... ', end='')
#     pc.delete_index(i)
#     print('Listo!')

# ? Insertar Embeddings en un Index de Pinecone

index_txt_name = '1984'
if index_txt_name not in pc.list_indexes().names():
    print(f'Creating Index {index_txt_name}')
    pc.create_index_for_model(
        name=index_txt_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"multilingual-e5-large",
            "field_map":{"text": "chunk_text"}
        }
    )
    print('Index Created!')
else:
    print(f'Index {index_txt_name} already exists!')

vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_txt_name)
# index = pc.Index(index_txt_name)
# print(index.describe_index_stats())

# ! Similarity Search
# question = "¿"
# result = vector_store.similarity_search(question)
# print(result)

# for r in result:
#     print(r.page_content)
#     print("-"*200)
# ? (Responde con fragmentos del texto, donde tiene lugar lo descrito en la question)

# ! Respondiendo en Natural Language with LLM

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# query = 'Responde solo a partir de la entrada dada: 1984 representa una sociedad distopica, intentado constituir una advertencia para la era moderna. ¿Sobre que nos esta advirtiendo el autor Orwell especificamente, y como lo logra?'
# answer = chain.invoke(query)
# print(answer)
# A partir de la entrada proporcionada, no se puede determinar específicamente sobre qué nos está advirtiendo George Orwell en su obra "1984". Para comprender mejor la 
# advertencia que el autor intenta transmitir en la novela distópica, es necesario analizar más a fondo el contenido y los temas desarrollados a lo largo de la historia. 
# Te recomendaría leer más allá de la introducción para obtener una comprensión más completa de la advertencia que Orwell intenta transmitir y cómo lo logra a lo largo de 
# la obra.

query = 'Responde solo a partir de la entrada dada: Conociendo el camino de Winston hacia su destruccion. ¿Donde encontramos, por primera vez, su perspectiva fatalista? ¿Es inevitable su derrota?'
answer = chain.invoke(query)
print(answer)
# En la entrada dada, encontramos por primera vez la perspectiva fatalista de Winston cuando reflexiona sobre cómo el proceso que lo llevará a su destrucción había comenzado 
# años atrás, desde un pensamiento secreto hasta acciones concretas. Aunque Winston sabe que eventualmente obedecerá a O'Brien, no está seguro de cuándo sucederá. Su perspectiva 
# fatalista radica en que siente que no puede escapar de este destino trazado desde hace tiempo. En este contexto, su derrota parece inevitable.

# ? Costo Embedding
# def print_embedding_cost(texts):
#     # https://platform.openai.com/docs/pricing / text-embedding-3-large
#     import tiktoken
#     enc = tiktoken.encoding_for_model('text-embedding-3-large')
#     total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
#     print(f'Totak tokens: {total_tokens}')
#     print(f'Embedding Cost in USD: {(total_tokens/1000000)*0.13:.6f}')
# print_embedding_cost(chunks)

# ? Vectores
# vectors = [[random.random() for _ in range(3072)] for v in range(5)]
# ids = list('abcde')

# index = pc.Index(index_name)
# index.upsert(vectors=zip(ids, vectors))
# print(index.fetch(ids=['c']))
# index.delete(ids=['b', 'c', 'd'])
# print(index.describe_index_stats())

# query_vectors = [random.random() for _ in range(3072)]
# print(index.query(vector=query_vectors, top_k=1, include_values=False))

# ? Namespaces
# index = pc.Index(index_name)

# vectors = [[random.random() for _ in range(3072)] for v in range(5)]
# ids = list('abcde')
# index.upsert(vectors=zip(ids, vectors))

# vectors = [[random.random() for _ in range(3072)] for v in range(3)]
# ids = list('xyz')
# index.upsert(vectors=zip(ids, vectors), namespace='primer-namespace')

# vectors = [[random.random() for _ in range(3072)] for v in range(2)]
# ids = list('qp')
# index.upsert(vectors=zip(ids, vectors), namespace='segundo-namespace')

# index.delete(ids='z', namespace='primer-namespace')
# print(index.fetch(ids=['z'], namespace='primer-namespace'))
