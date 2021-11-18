#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import matplotlib.pyplot as plt

multistream_data = pandas.read_csv("results.csv")

multistream_means = multistream_data.mean()

fig, ax = plt.subplots(figsize=(20,8), dpi=150)
ax.grid(axis='y')
ax.bar(x=multistream_means.index, height=multistream_means)
plt.ylabel("Completion Time (ms)")
plt.xlabel("# Streams Utilized")
plt.title("Mean Completion Times for Matrix Multiplication in Series Using Streams")
plt.show()

