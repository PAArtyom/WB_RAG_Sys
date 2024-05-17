from fastapi import FastAPI, Form, Response, Request, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from loguru import logger

import pandas as pd
import numpy as np

from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory

from sklearn.metrics.pairwise import cosine_similarity

import time
import json
import os
import wget

from llama_cpp import Llama
from Embed import HFEmbeddings

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

embedding = HFEmbeddings(model_name="cointegrated/LaBSE-en-ru")

# Сохраняем чанки и вопросы+ответы в списке и читаем кэш
qas = pd.read_csv('data/preprocessed_qa.csv').dropna()
docs = pd.read_csv('data/docs.csv')["chunk"].to_list()

qas['quesans'] = qas['question'] + "<nt>" + qas['answer']
quesans = qas['quesans'].tolist()
qas['vec_question'] = embedding.embed_documents(qas['question'])
logger.info("Вопросы векторизированы")

cache = pd.read_csv('data/cache.csv')
cache['vec_question'] = embedding.embed_documents(cache['question'])


# Инициализация ретривера
db = FAISS.load_local('db/', embedding, allow_dangerous_deserialization=True)
faiss_retriever = db.as_retriever(search_kwargs={"k": 3})

bm25_docs_retriever = BM25Retriever.from_texts(docs)
bm25_docs_retriever.k = 2

bm25_quesans_retriever = BM25Retriever.from_texts(quesans)
bm25_quesans_retriever.k = 2

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_docs_retriever, faiss_retriever,
                                                   bm25_quesans_retriever],
                                       weights=[0.33, 0.33, 0.33])
logger.info("Ретриверы собраны")

model_path = "model/model-q2_k.gguf"
url = 'https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q2_K.gguf'

if not os.path.exists(model_path):
    logger.info("Модель не обнаружена, скачиваем...")
    wget.download(url, model_path)
    logger.info("Модель скачана")

llm = LlamaCpp(model_path=model_path,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.5,
    repeat_penalty=1,
    n_gpu_layers=-1
              )

logger.info("Модель инициализирована")

prompt_template = """
Ты — автоматизированный виртуальный помощник техподдержки Wildberries.
Твоя задача — обеспечить клиентов точной, полезной информацией, отвечая на их вопросы.
Используя контекстную информацию ниже составляй корректный и краткий ответ. Если ты не знаешь
ответ, то просто отвечай "Я не уверен", не пытайся придумать свой собственный.

Контекст: {context}
Вопрос: {question}

Убедись, что ты вежливо обращаешься с клиентами и предоставляешь только полезные
инструкции или ответы. 

Ответ:
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])
chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=ensemble_retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs,
                                     output_key='result',
                                     memory=ConversationBufferMemory(
                                         memory_key='context',
                                         output_key='result'),
                                     verbose=True)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def vectorize_and_compare(input_text: str, df_base: pd.DataFrame, df_cache: pd.DataFrame, qas_base: pd.Series,
                          cache_base: pd.Series, threshold: float = 0.9) -> pd.DataFrame:
    # Векторизация входного текста
    input_vector = np.array(embedding.embed_query(input_text))

    # Преобразование векторов в массивы numpy
    qas_vectors = np.array(qas_base.tolist())
    cache_vectors = np.array(cache_base.tolist())

    # Проверка размеров векторов
    if qas_vectors.shape[1] != cache_vectors.shape[1]:
        raise ValueError(
            f"Размеры векторов не совпадают: qas_base имеет размер {qas_vectors.shape[1]}, а cache_base - {cache_vectors.shape[1]}")

    if qas_vectors.shape[1] != input_vector.shape[0]:
        raise ValueError(
            f"Размеры векторов не совпадают: вектор запроса имеет размер {input_vector.shape[0]}, а векторы вопросов - {qas_vectors.shape[1]}")
    # Объединение векторов базы и кэша
    all_vectors = np.vstack((qas_vectors, cache_vectors))

    # Вычисление косинусного сходства
    similarities = np.dot(all_vectors, input_vector) / (
                np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(input_vector))
    # Поиск индекса самого похожего вопроса
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]

    if max_similarity < threshold:
        return pd.DataFrame()  # Возвращает пустой DataFrame, если сходство меньше порога

    # Определение, находится ли наиболее похожий вопрос в базе или в кэше
    if max_similarity_idx < len(qas_base):
        similar_text = df_base.iloc[max_similarity_idx]
    else:
        similar_text = df_cache.iloc[max_similarity_idx - len(qas_base)]

    return similar_text


@app.post("/get_response")
def get_response(query: str = Form(...)):

    start_time = time.time()

    # Проверяем, есть ли вопрос в кэше, если есть, то сразу выдаём ответ
    if query in cache['question'].values:
        answer = cache[cache['question'] == query]['answer'].values[0]
        response_time = time.time() - start_time  # Время выполнения
        return JSONResponse(content={"answer": answer, "response_time": response_time})

    if query in qas['question'].values:
        answer = qas[qas['question'] == query]['answer'].values[0]
        response_time = time.time() - start_time  # Время выполнения
        return JSONResponse(content={"answer": answer, "response_time": response_time})

    # Проверка семантической схожести с существующими вопросами
    similar_question = vectorize_and_compare(query, qas, cache, qas['vec_question'], cache['vec_question'],
                                             threshold=0.82)

    if not similar_question.empty:
        answer = similar_question['answer']
        response_time = time.time() - start_time  # Время выполнения
        return JSONResponse(content={"answer": answer, "response_time": response_time})

    # Если вопроса нет в кэше, обращаемся к модели
    response = qa(query)  # Ваше обращение к модели
    answer = response['result']
    response_time = time.time() - start_time  # Время выполнения

    # Сохраняем ответ в кэше, если ранее его не было
    if query not in cache['question'].values:
        cache.loc[len(cache)] = {'question': query, 'answer': answer, 'context': response['context'],
                                 'source_documents': response['source_documents']}
        cache.to_csv("data/cache.csv", index=False)

    return JSONResponse(content={"answer": answer, "response_time": response_time})
