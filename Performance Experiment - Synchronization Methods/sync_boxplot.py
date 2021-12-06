#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas
import matplotlib.pyplot as plt

sync_data = pandas.read_csv("results.csv")


# Create a plot with a sub-boxplot for each transfer type
fig, axes = plt.subplots(1,3, figsize=(3,6), dpi=120, sharey=True)

axes[0].set_ylabel("Completion Time (s)")

# Create each boxplot in its own dedicated plot and color
colors = ["orange", "purple", "red"]
for i, col in enumerate(sync_data.columns):
    axes[i].boxplot(x=sync_data[col], patch_artist=True, boxprops=dict(facecolor=colors[i]), medianprops=dict(color='cyan'), showfliers=False)
    axes[i].set_xlabel(col)
    axes[i].set_xticks([])
    
fig.patch.set_facecolor("white")
plt.suptitle("Synchronization Method Latencies")
    
plt.show()


# In[ ]:




