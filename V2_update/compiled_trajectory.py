#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import librosa


# In[2]:


def mouse_columns_x(n):
    mice_x = [f'M{i}_center_x' for i in range(1, n+1)]
    return mice_x

def mouse_columns_y(n):
    mice_y = [f'M{i}_center_y' for i in range(1, n+1)]
    return mice_y


# In[3]:


def path_data(n):

   
    for i in range (1, n+1):
        column_x = f'M{i}_center_x'
        Mi_x = globals()[f'M{i}_escape_only']['center_x'].reset_index(drop=True)
        mouse_id[column_x] = pd.Series(Mi_x, index = mouse_id.index)

        
    for i in range (1,n+1):
        column_y = f'M{i}_center_y'
        Mi_y = globals()[f'M{i}_escape_only']['center_y'].reset_index(drop=True)
        mouse_id[column_y] = pd.Series(Mi_y, index = mouse_id.index)
        
    return mouse_id


# In[4]:


def summarize_path(mice_x, mice_y, df):
    for i in range(len(mice)):
        column_name = mice_x[i]
        df[column_name] = normalize(df[column_name])
        
    for i in range(len(mice)):
        column_name = mice_y[i]
        df[column_name] = normalize(df[column_name])


# In[5]:


def plot_trajectory(df, mice_x, mice_y):    
    fig = plt.figure()  #create figure to fill in
    ax = plt.axes()
    
    for i in range(len(mice_x)):
        column_x = mice_x[i]
        xi = df[column_x]
        xf = df[column_x].iloc[-1] #final x coordinate
        xo = df[column_x].iloc[-2] #second to last x coordinate
        
    for i in range(len(mice_y)):
        column_y = mice_y[i]
        yi = df[column_y]
        yf = df[column_y].iloc[-1] #final y coordinate
        yo = df[column_y].iloc[-2] #second to last y coordinate
        
        
    plt.plot(xi,yi)
    ax.plot(x,y, color = 'blue', linewidth = 1)

    #add an arrow to show mouse's direction
    #add an arrow to show mouse's direction



    plt.arrow(xo, yo, xf-xo, yf-yo, head_width = 1, head_length = 1, fc = 'blue', ec = "none")
    ax.set_title('Trajectory during stimulus')  #would be after stimulus
    ax.set_xlabel('x-position (cm)', fontsize=12)
    ax.set_ylabel('y-position (cm)', fontsize=12)

    plt.axis('off')

        
    return plt


# In[ ]:




