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


#%run graphing_function_recent.py
from graphing_function_recent import total_seconds, likelihood_check, dataframe_ranges, convert_time, audio_timing, time_set_zero, convert_data, single_mouse_data, displacement, speed, angle, angle_speed, trajectory, head_angle_trajectory_figure, speed_figure, displacement_figure, head_angle_figure, linearity_ratio, angle_speed_figure, escape_time, filter_data, shelter_rotation

from V3_M1 import coord_scaled as M1_coord_scaled, escape_only as M1_escape_only, four_second as M1_four_second, six_second as M1_six_second, long_range as M1_long_range, initial_displacement as M1_intial_displacement, total_distance as M1_total_distance

from V3_M2 import coord_scaled as M2_coord_scaled, escape_only as M2_escape_only, four_second as M2_four_second, six_second as M2_six_second, long_range as M2_long_range, initial_displacement as M2_intial_displacement, total_distance as M2_total_distance

from V3_M3 import coord_scaled as M3_coord_scaled, escape_only as M3_escape_only, four_second as M3_four_second, six_second as M3_six_second, long_range as M3_long_range, initial_displacement as M3_intial_displacement, total_distance as M3_total_distance

from V3_M4 import coord_scaled as M4_coord_scaled, escape_only as M4_escape_only, four_second as M4_four_second, six_second as M4_six_second, long_range as M4_long_range, initial_displacement as M4_intial_displacement, total_distance as M4_total_distance

from V3_M7 import coord_scaled as M5_coord_scaled, escape_only as M5_escape_only, four_second as M5_four_second, six_second as M5_six_second, long_range as M5_long_range, initial_displacement as M5_intial_displacement, total_distance as M5_total_distance

from V3_M8 import coord_scaled as M6_coord_scaled, escape_only as M6_escape_only, four_second as M6_four_second, six_second as M6_six_second, long_range as M6_long_range, initial_displacement as M6_intial_displacement, total_distance as M6_total_distance

from V3_M9 import coord_scaled as M7_coord_scaled, escape_only as M7_escape_only, four_second as M7_four_second, six_second as M7_six_second, long_range as M7_long_range, initial_displacement as M7_intial_displacement, total_distance as M7_total_distance

from V3_M10 import coord_scaled as M8_coord_scaled, escape_only as M8_escape_only, four_second as M8_four_second, six_second as M8_six_second, long_range as M8_long_range, initial_displacement as M8_intial_displacement, total_distance as M8_total_distance


# In[ ]:


#number of dataframes for parameter:
#return list M1, M2, etc.

#string determines what column you look at 
def mouse_columns(n, string):
    mice = [f'M{i}_' + string for i in range(1, n+1)]
    return mice


# In[ ]:


#create a new dataframe with interpolated displacement, speed, or head angle (choose string)
def interpolate_data(start, stop, step, string, df):
    interpolate = pd.DataFrame()
    
    time = np.arange(start,stop,step)
    interpolate['time'] = time
    
    #interpolate displacement data
    data_raw = df[string]
    time_raw = df['time_set']
    
    displacement_inter = np.interp(time, time_raw, data_raw)
    
    return displacement_inter


# In[ ]:


#returns a dataframe with interpolated values 
#column_string determines which paramater you look at
#df_string determines which dataframe you look at 

def interpolate_all(n, start, stop, step, column_string, df_string):
    compiled_name = column_string + '_interpolated'
    compiled_name = pd.DataFrame()
    
    for i in range(1, n+1):
        name = f'M{i}_' + df_string #ex: look at just four_second
        df = globals()[name]
        
        interpolated_data = interpolate_data(start, stop, step, column_string, df)
        compiled_name[name + '_interpolated'] = interpolated_data
    
    time = np.arange(start,stop,step)  
    compiled_name['time'] = time
    
    return compiled_name


# In[ ]:


#return normalized df
def normalize(df):
    min_value = df.min()
    max_value = df.max()
    range_value = max_value - min_value
    normalized = (df - min_value) / range_value
    return normalized


# In[ ]:


#find mean and upper and lower limit for SEM
def summarize_data(n, string, df):
    mice = mouse_columns(n, string)
    
    for i in range(len(mice)):
        column_name = mice[i]
        df[column_name] = normalize(df[column_name])
        
    df['average'] = df[mice].mean(axis = 1)
    SEM = df[mice].sem(axis = 1)
    df['SEM_up'] = df['average'] + SEM
    df['SEM_down'] = df['average'] - SEM


# In[ ]:


#plot average + SEM
def plot_compiled_displacement(df, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    y = df['average']
    y_up = df['SEM_up']
    y_down = df['SEM_down']

    ax.plot(x,y)
    ax.plot(x,y_up, color = 'none')
    ax.plot(x,y_down, color = 'none')
    plt.fill_between(x,y_up, y_down, color = 'blue', alpha = .05)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Displacement (cm)') #convert to cm - cm/s
    ax.set_title('Average displacement from shelter after stimulus onset')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)
    
    plt.xlim(min(x), max(x))

    # Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


def plot_all_displacement(df, mice, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    
    for i in range(len(mice)):
        column_name = mice[i]
        yi = df[column_name]
        plt.plot(x,yi)

    ax.set_xlabel('Time from stimulus (s)')
    ax.set_ylabel('Displacement from shelter (cm)') #convert to cm - cm/s
    ax.set_title('Displacement from shelter during stimulus presentation')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)

    plt.xlim(min(x), max(x))

    #Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


#plot compiled average and SEM for head angle
def plot_compiled_angle(df, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    y = df['average']
    y_up = df['SEM_up']
    y_down = df['SEM_down']

    ax.plot(x,y)
    ax.plot(x,y_up, color = 'none')
    ax.plot(x,y_down, color = 'none')
    plt.fill_between(x,y_up, y_down, color = 'blue', alpha = .05)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Head angle (°)') #convert to cm - cm/s
    ax.set_title('Head angle from shelter after stimulus onset')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)
        
    plt.xlim(min(x), max(x))

    # Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


#plot all for angle
def plot_all_angle(df, mice, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    
    for i in range(len(mice)):
        column_name = mice[i]
        yi = df[column_name]
        plt.plot(x,yi)

    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Head angle (°)') #convert to cm - cm/s
    ax.set_title('Head angle from shelter after stimulus onset')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)

        
    plt.xlim(min(x), max(x))

    #Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


#plot average + SEM for speed
def plot_compiled_speed(df, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    y = df['average']
    y_up = df['SEM_up']
    y_down = df['SEM_down']

    ax.plot(x,y)
    ax.plot(x,y_up, color = 'none')
    ax.plot(x,y_down, color = 'none')
    plt.fill_between(x,y_up, y_down, color = 'blue', alpha = .05)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Speed (cm/s)') #convert to cm - cm/s
    ax.set_title('Average speed after stimulus onset')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)
        
    plt.xlim(min(x), max(x))

    # Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


#plot all speed
def plot_all_speed(df, mice, stop, length):
    plt.figure()
    ax = plt.axes()

    #graph of displacement vs time after stimulus plt.figure() ax = plt.axes()

    x = df['time']
    
    for i in range(len(mice)):
        column_name = mice[i]
        yi = df[column_name]
        plt.plot(x,yi)

    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Speed (cm/s)') #convert to cm - cm/s
    ax.set_title('Speed after stimulus onset')

    if length<8:
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Vertical Line')
    else:
        plt.axvspan(0, stop, color = 'b', alpha =.08)

        
    plt.xlim(min(x), max(x))

    #Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:


#plot graph of total initial distance vs displacement of mouse 
def linearity_ratio(mice_distance, mice_initial_displacement, df):
    plt.figure()
    ax = plt.axes()
    
    for i in range(len(mice_distance)):
        column_name_1 = mice_total_distance[i]
        yi = df[column_name_2]
        
        column_name_2 = mice_initial_displacement[i]
        xi = df[column_name_2]
        plt.plot(xi,yi)
        
    ax.set_xlabel('Total distance from shelter at stimulus initiation (cm)')
    ax.set_ylabel('Total displacement of mouse during escape (cm)')
    ax.set_title('Linearity of escape')
    
    plt.xlim(min(x), max(x))

    #Remove the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# In[ ]:




