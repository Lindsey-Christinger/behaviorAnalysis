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

from graphing_function_recent import total_seconds, likelihood_check, dataframe_ranges, convert_time, audio_timing, time_set_zero, convert_data, single_mouse_data, displacement, speed, angle, angle_speed, trajectory, head_angle_trajectory_figure, speed_figure, displacement_figure, head_angle_figure, linearity_ratio, angle_speed_figure, escape_time, filter_data, shelter_rotation


# In[2]:


#Use this for all adjustable values 
#input file paths here (include .csv):
video_path = r'\\10.25.16.5\yaochen\Active\DLC\Final_Models\Lindsey\escape_behavior_2-lindsey_christinger-2024-06-13_v1\V3_Tested2\coords_csv\CoordV3_M10_P2_6_22_24_videoDLC_resnet50_escape_behavior_2Jun13shuffle1_1030000_filtered_day.csv'
time_path = r'\\10.25.16.5\yaochen\Active\Lindsey\set_up_testing\V3_updated\Timestamp\M10_P2_6_22_24_timestamp0.csv'
audio_path = r'\\10.25.16.5\yaochen\Active\Lindsey\set_up_testing\V3_updated\Audio\M10_P2_6_22_24_audio.wav'

#input coordinates for the shelter and platform 
north_pixel = 26
south_pixel = 462
east_pixel = 570
west_pixel = 101
shelter_x_pixel = 140
shelter_y_pixel = 258

#input info from time_and_audio info notebook (either from audacity or bonsai)
stimulus_start = 322.266 #stimulus start time
stimulus_end = 329.881 #stimulus end time
escape_done = 324 #time mouse reaches shelter location


# In[3]:


#manually decide?
escape_response = True
freezing_response = False


# In[4]:


#read in DLC data, convert to cm, center platform at 0,0
shelter_x, shelter_y, x_diam, y_diam, coord_scaled = single_mouse_data(north_pixel, south_pixel, east_pixel, west_pixel, shelter_x_pixel, shelter_y_pixel, video_path)


# In[5]:


#replace any head, nose, or center points with likelihood < .9 with NA
filter_data(.8, coord_scaled)


# In[6]:


#coord_filtered


# In[7]:


#convert timestamps to zeroed seconds
convert_time(time_path, coord_scaled)


# In[8]:


#REPLACE ONCE WE HAVE STIMULUS TIMESTAMPS
#returns start and stop times of stimulus.
start, stimulus_end = audio_timing(stimulus_start, stimulus_end, audio_path, coord_scaled['time'].iloc[-1])


# In[9]:


#find time stimulus ends and mouse reaches shelter(escape) if stimulus is set to 0s
end, escape = time_set_zero(start, stimulus_end, escape_done, coord_scaled)


# In[10]:


#calculate displacement as 'displacement'
displacement(shelter_x, shelter_y, coord_scaled)


# In[11]:


#calculate speed as 'speed' and change in displacement as 'delta_vector'
total_distance = speed(coord_scaled)


# In[12]:


#calculate head angle as 'head_angle'
angle(shelter_x, shelter_y, coord_scaled)


# In[13]:


#calculate head angle speed as 'angle_speed'
angle_speed(coord_scaled)


# In[14]:


#create dataframes isolated to 4 ranges: just shelter reaching, 3 seconds before and 6 after, 2 before and 4 after, and 1 where you can input your own parameters
#return 4 dataframes 
escape_only, four_second, six_second, long_range = dataframe_ranges(-15, 30, escape,coord_scaled)


# In[15]:


trajectory(shelter_x, shelter_y, x_diam, y_diam, escape_only)


# In[16]:


head_angle_trajectory_figure(shelter_x, shelter_y, x_diam, y_diam, escape_only)


# In[17]:


speed_figure(four_second['time_set'].iloc[-1],4, four_second)


# In[18]:


speed_figure(six_second['time_set'].iloc[-1],4, six_second)


# In[19]:


speed_figure(end,9, long_range)


# In[20]:


displacement_figure(four_second['time_set'].iloc[-1],4, four_second)


# In[21]:


displacement_figure(six_second['time_set'].iloc[-1],4, six_second)


# In[22]:


displacement_figure(end,9, long_range)


# In[23]:


head_angle_figure(four_second['time_set'].iloc[-1],4, four_second)


# In[24]:


head_angle_figure(six_second['time_set'].iloc[-1],4, six_second)


# In[25]:


angle_speed_figure(four_second['time_set'].iloc[-1], 4,four_second)


# In[26]:


ratio, initial_displacement, total_distance = linearity_ratio(escape_only)


# In[27]:


# In[28]:


likelihood_data, likelihood_minus_nose, likelihood_center = likelihood_check(0, 8, .9, coord_scaled)


# In[29]:


likelihood_data #need to adjust time frame to be useful 


# In[ ]:




