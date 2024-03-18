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


#!pip install openai==0.28.1


# In[6]:


import streamlit as st
from langchain.llms import OpenAI
st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

import os
os.environ["OPENAI_API_KEY"] = "sk-"  #openai 키 입력

def generate_response(input_text):  #llm이 답변 생성
    llm = OpenAI(model_name='gpt-4-0314', temperature=0)
    st.info(llm(input_text))

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?') #첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')
    generate_response(text)


# In[ ]:




