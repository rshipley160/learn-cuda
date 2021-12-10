#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas
import matplotlib.pyplot as plt

completionTimes = pandas.read_csv("results.csv", header=None).T
header = completionTimes.iloc[0]
completionTimes = completionTimes[1:]
header = header.apply(int)
completionTimes.columns = header

fig = plt.figure(dpi=120)
fig.patch.set_facecolor("white")

completionTimes.median().plot(kind="bar")
plt.ylim(900, 1500)
plt.xticks(rotation=0)
plt.xlabel("# of built in solver iterations")
plt.ylabel("Completion Time (ms)")
plt.title("Completion Time VS # Built-in Iterations")
plt.show()


# In[ ]:




