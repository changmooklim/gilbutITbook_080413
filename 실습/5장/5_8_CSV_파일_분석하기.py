#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain-experimental==0.0.47')


# In[2]:


get_ipython().system('pip install tabulate==0.9.0')


# In[3]:


get_ipython().system('pip install pandas==2.1.4')


# In[4]:


get_ipython().system('pip install openai==1.6.1')


# In[5]:


import pandas as pd #파이썬 언어로 작성된 데이터를 분석 및 조작하기 위한 라이브러리

#csv 파일을 데이터프레임으로 가져오기
df = pd.read_csv('e:/data/booksv_02.csv') #booksv_02.csv 파일이 위치한 경로 지정
df.head()


# In[6]:


from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
os.environ["OPENAI_API_KEY"] = "sk-" #openai 키 입력

# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-3.5-turbo-16k-0613'),        #gpt-3.5-turbo 사용
    df,                                    #데이터가 담긴 곳
    verbose=False,                          #추론 과정을 출력하지 않음
    agent_type=AgentType.OPENAI_FUNCTIONS, 
)


# In[7]:


agent.run('어떤 제품의 ratings_count가 제일 높아?')


# In[8]:


agent.run('가장 최근에 출간된 책은?')


# In[ ]:




