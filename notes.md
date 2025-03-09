# Langchain-Practice:

## Conceptos Principales:

- LangChain: marco de código abierto que ayuda a los desarrolladores 
a crear aplicaciones basadas en modelos de lenguaje de gran tamaño (LLM)

- LLM (Large Language Model): modelo de aprendizaje profundo que pueden 
generar respuestas a preguntas o crear imágenes. 

- Prompt: instrucción, pregunta o texto que se le da a una inteligencia 
artificial (IA) para que genere un resultado. Cuando se usa 
plantillas para prompts (Prompts Templates), se crea una estructura 
que ayuda a incluir todos los elementos importantes de manera organizada y coherente.

- Chains: 
    - Simple Chain: secuencia lineal de pasos donde cada etapa procesa 
    la información de forma independiente y sin depender directamente 
    del resultado del paso anterior. Se utiliza para generar respuestas 
    complejas de manera directa, ensamblando cada etapa en un flujo.

    - Sequential Chains: conjuntos de módulos interconectados en los que 
    la salida de un paso se convierte en la entrada del siguiente. 
    Esto permite que la información se refine progresivamente, 
    construyendo respuestas complejas paso a paso mediante la 
    dependencia entre etapas

- Agentes ReAct: modelos de IA que combinan razonamiento y acción de 
forma intercalada, permitiendo pensar y actuar paso a paso para 
resolver tareas complejas. También se pueden habilitar "Tools" que 
ayuden en la resolución

- Vectores: representaciones numéricas (listas de números) que codifican 
información de datos, facilitando su procesamiento y manipulación por 
algoritmos.

- Embedding: técnica que transforma datos discretos (como palabras o 
ítems) en vectores densos en un espacio continuo, de forma que 
elementos con significados o características similares queden próximos 
entre sí. Técnica de procesamiento de lenguaje natural que convierte 
el lenguaje humano en vectores matemáticos.

- Pinecone: base de datos vectorial en la nube que permite almacenar, 
buscar y gestionar vectores de alta dimensión. Se utiliza para 
aplicaciones de machine learning, como la búsqueda de imágenes y 
texto, la generación de texto, la detección de anomalías y fraudes, 
y más. 