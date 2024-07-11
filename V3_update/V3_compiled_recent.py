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


from compiled_functions_recent import mouse_columns, interpolate_data, interpolate_all, normalize, summarize_data, plot_compiled_displacement, plot_all_displacement, plot_compiled_speed, plot_all_speed, plot_compiled_angle, plot_all_angle

from compiled_trajectory import mouse_columns_x, mouse_columns_y, path_data, summarize_path, plot_trajectory


# In[3]:


from graphing_function_recent import total_seconds, likelihood_check, dataframe_ranges, convert_time, audio_timing, time_set_zero, convert_data, single_mouse_data, displacement, speed, angle, angle_speed, trajectory, head_angle_trajectory_figure, speed_figure, displacement_figure, head_angle_figure, linearity_ratio, angle_speed_figure, escape_time, filter_data, shelter_rotation

from V3_M1 import coord_scaled as M1_coord_scaled, escape_only as M1_escape_only, four_second as M1_four_second, six_second as M1_six_second, long_range as M1_long_range, initial_displacement as M1_intial_displacement, total_distance as M1_total_distance, shelter_x as M1_shelter_x, shelter_y as M1_shelter_y

from V3_M2 import coord_scaled as M2_coord_scaled, escape_only as M2_escape_only, four_second as M2_four_second, six_second as M2_six_second, long_range as M2_long_range, initial_displacement as M2_intial_displacement, total_distance as M2_total_distance, shelter_x as M2_shelter_x, shelter_y as M2_shelter_y

from V3_M3 import coord_scaled as M3_coord_scaled, escape_only as M3_escape_only, four_second as M3_four_second, six_second as M3_six_second, long_range as M3_long_range, initial_displacement as M3_intial_displacement, total_distance as M3_total_distance, shelter_x as M3_shelter_x, shelter_y as M3_shelter_y

from V3_M4 import coord_scaled as M4_coord_scaled, escape_only as M4_escape_only, four_second as M4_four_second, six_second as M4_six_second, long_range as M4_long_range, initial_displacement as M4_intial_displacement, total_distance as M4_total_distance, shelter_x as M4_shelter_x, shelter_y as M4_shelter_y

from V3_M7 import coord_scaled as M5_coord_scaled, escape_only as M5_escape_only, four_second as M5_four_second, six_second as M5_six_second, long_range as M5_long_range, initial_displacement as M5_intial_displacement, total_distance as M5_total_distance, shelter_x as M5_shelter_x, shelter_y as M5_shelter_y

from V3_M8 import coord_scaled as M6_coord_scaled, escape_only as M6_escape_only, four_second as M6_four_second, six_second as M6_six_second, long_range as M6_long_range, initial_displacement as M6_intial_displacement, total_distance as M6_total_distance, shelter_x as M6_shelter_x, shelter_y as M6_shelter_y

from V3_M9 import coord_scaled as M7_coord_scaled, escape_only as M7_escape_only, four_second as M7_four_second, six_second as M7_six_second, long_range as M7_long_range, initial_displacement as M7_intial_displacement, total_distance as M7_total_distance, shelter_x as M7_shelter_x, shelter_y as M7_shelter_y

from V3_M10 import coord_scaled as M8_coord_scaled, escape_only as M8_escape_only, four_second as M8_four_second, six_second as M8_six_second, long_range as M8_long_range, initial_displacement as M8_intial_displacement, total_distance as M8_total_distance, shelter_x as M8_shelter_x, shelter_y as M8_shelter_y


# In[4]:


#interpolate data, combine into one dataframe, and find mean/SEM - input number of mice, start, ,stop, step, column of interest, dataframe of interes
#displacement 
displacement_inter_four = interpolate_all(8, -2, 4, .04, 'displacement', 'four_second')
summarize_data(8, 'four_second_interpolated', displacement_inter_four)

displacement_inter_six = interpolate_all(8, -3, 6, .04, 'displacement', 'six_second')
summarize_data(8, 'six_second_interpolated', displacement_inter_six)

#speed
speed_inter_four = interpolate_all(8, -2, 4, .04, 'speed', 'four_second')
summarize_data(8, 'four_second_interpolated', speed_inter_four)

speed_inter_six = interpolate_all(8, -3, 6, .04, 'speed', 'six_second')
summarize_data(8, 'six_second_interpolated', speed_inter_six)

#head angle
angle_inter_four = interpolate_all(8, -2, 4, .04, 'head_angle', 'four_second')
summarize_data(8, 'four_second_interpolated', angle_inter_four)

angle_inter_six = interpolate_all(8, -3, 6, .04, 'head_angle', 'six_second')
summarize_data(8, 'six_second_interpolated', angle_inter_six)


# In[5]:


plot_compiled_displacement(displacement_inter_four,7.5, 4)


# In[6]:


plot_compiled_displacement(displacement_inter_six, 7.5,4)


# In[7]:


plot_compiled_speed(speed_inter_four,7.5, 4)


# In[8]:


plot_compiled_speed(speed_inter_six, 7.5,4)


# In[9]:


plot_compiled_angle(angle_inter_four,7.5, 4)


# In[10]:


plot_compiled_angle(angle_inter_six, 7.5,4)


# In[11]:


plot_all_displacement(displacement_inter_four, mouse_columns(8, 'four_second_interpolated'), 7, 4)


# In[29]:


plt.figure()
ax=plt.axes()

x = displacement_inter_four['time']
y = displacement_inter_four['average']
y_up = displacement_inter_four['SEM_up']
y_down = displacement_inter_four['SEM_down']

ax.plot(x,y, label = 'displacement')
ax.plot(x,y_up, color = 'none')
ax.plot(x,y_down, color = 'none')
plt.fill_between(x,y_up, y_down, color = 'blue', alpha = .05)
ax.set_xlabel('Time from stimulus onset (s)')
ax.set_ylabel('Displacement (cm) and speed (cm/s)') #convert to cm - cm/s
ax.set_title('Average displacement and speed after stimulus onset')
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

xs = speed_inter_four['time']
ys = speed_inter_four['average']
y_ups = speed_inter_four['SEM_up']
y_downs = speed_inter_four['SEM_down']

ax.plot(xs,ys, color='red', label = 'speed')
ax.plot(xs,y_ups, color = 'none')
ax.plot(xs,y_downs, color = 'none')
plt.fill_between(xs,y_ups, y_downs, color = 'red', alpha = .05)

xa = angle_inter_four['time']
ya = angle_inter_four['average']
y_upa = angle_inter_four['SEM_up']
y_downa = angle_inter_four['SEM_down']

ax.plot(xa,ya, color='green', label = 'head angle')
ax.plot(xa,y_upa, color = 'none')
ax.plot(xa,y_downa, color = 'none')
plt.fill_between(xa,y_upa, y_downa, color = 'green', alpha = .05)

plt.legend(loc = 'upper right')


# In[ ]:




