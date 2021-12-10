#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import matplotlib.pyplot as plt

execTimes = pandas.read_csv("results.csv")

# Create a plot with a sub-boxplot for each transfer type
fig, axes = plt.subplots(1,4, figsize=(10,4), dpi=120,)

axes[0].set_ylabel("Completion Time (s)")

# Create each boxplot in its own dedicated plot and color
colors = ["orange", "purple", "red","green"]
for i, col in enumerate(execTimes.columns):
    axes[i].boxplot(x=execTimes[col], patch_artist=True, boxprops=dict(facecolor=colors[i]), medianprops=dict(color='cyan'), showfliers=False)
    axes[i].set_xlabel(col)
    axes[i].set_xticks([])
    
axes[2].set_ylim(30, 33)
axes[3].set_ylim(30, 33)
fig.patch.set_facecolor("white")
plt.suptitle("Completion Time for Various Execution Methods")
plt.subplots_adjust(wspace=1)
    
plt.show()

