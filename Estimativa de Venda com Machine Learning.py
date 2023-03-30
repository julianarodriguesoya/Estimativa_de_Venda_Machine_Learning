#!/usr/bin/env python
# coding: utf-8

# # Estimativa de Venda com Machine Learning
# 

# # O objetivo deste notebook é a prática do conteúdo apresentado pelo curso da hastag treinamentos para aprendizagem do uso de Machine Learning.
# 

# # Pergunta a ser respondida:Será investido 75k em marketing, qual deve ser o estoque enviado para a loja?

# # Importando a base de vendas

# In[26]:


#bibliotecas usadas

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[27]:


# importar a base de vendas
base = pd.read_excel ("Investimento_x_Venda.xlsx")


# In[28]:


# Exibindo as 5 primeiras linhas
base.head()


# In[29]:


# Visualizando de forma gráfica essas informações
plt.scatter(base["Investimento em marketing"], base ["Venda Qtd"])
plt.show()


# # Traçando uma reta passando por estes pontos

# In[30]:


plt.scatter(base["Investimento em marketing"], base  ["Venda Qtd"])
x0 = base ["Investimento em marketing"] [0]
y0 = base ["Venda Qtd"] [0]
x1 = base ["Investimento em marketing"][6]
y1 = base ["Venda Qtd"] [6]
plt.plot([x0,x1],[y0,y1],"r")
plt.show()


# # Usando a equação da reta para determinar a venda
# # y=ax+b

# In[23]:


def EncontraY(x_reta,y_reta, x):
    a = (y_reta[1] - y_reta[0])/(x_reta[1] - x_reta[0])
    b = y_reta[1] - a*x_reta[1]
    y = a*x + b
    return y


# In[24]:


EncontraY([x0,x1],[y0,y1],75)


# # Descobrindo a Venda usando Machine Learning
# 

# In[31]:


from sklearn import linear_model


# In[32]:


reg = linear_model.LinearRegression()


# In[33]:


reg.fit(base["Investimento em marketing"].values.reshape(-1, 1),base["Venda Qtd"])


# In[34]:


reg.coef_


# In[35]:


reg.intercept_


# In[36]:


plt.scatter(base["Investimento em marketing"],base["Venda Qtd"])
x = np.array(base["Investimento em marketing"])
y = reg.intercept_ + x*reg.coef_
plt.plot(x,y,"r")
plt.show()


# In[37]:


reg.predict([[75]])


# In[38]:


plt.scatter(base["Investimento em marketing"],base["Venda Qtd"])
plt.scatter(75,reg.predict([[75]])[0],color="k")
x = np.array(base["Investimento em marketing"])
y = reg.intercept_ + x*reg.coef_
plt.plot(x,y,"r")
plt.show()


# In[ ]:




