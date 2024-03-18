#!/usr/bin/env python
# coding: utf-8

# In[1]:


#langchain==0.0.350에서 dependency 오류가 발생하여 기존 langchanin 삭제
#!pip3 uninstall -y langchain


# In[2]:


#langchain==0.0.350에서 dependency 오류가 발생하여 langchanin 최신 버전으로 설치
#!pip install langchain


# In[3]:


#langchain-community 역시 재설치
#!pip install -U langchain-community


# In[4]:


#!pip install streamlit==1.29.0


# In[5]:


#!pip install PyPDF2==3.0.1


# In[6]:


#!pip install sentence-transformers==2.2.2


# In[7]:


import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def process_text(text): 
#CharacterTextSplitter를 사용하여 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    #임베딩 처리(벡터 변환), 임베딩은 HuggingFaceEmbeddings 모델을 사용합니다.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():  #streamlit을 이용한 웹사이트 생성
    st.title("📄PDF 요약하기")
    st.divider()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-" #openai api 키 입력
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""   # 텍스트 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."  # LLM에 PDF파일 요약 요청

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('--요약 결과--:')
            st.write(response)

if __name__ == '__main__':
    main()


# In[ ]:




