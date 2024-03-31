#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Functions

# In[ ]:


def plot_time_series(df_graphing, column_name):
    plt.figure(figsize=(12, 8))

    # Define colors for the different feeding frequencies
    colors = {1: 'blue', 2: 'green'}

    # Track the feeding frequencies that have been added to the legend
    added_legend = set()

    for (feeding_freq, date), group in df_graphing.groupby(['Feeding_Freq', 'Date']):
        group_sorted = group.sort_index()  # Sort the group by the index (TimeNumeric)

        # Check if this feeding frequency has already been added to the legend
        if feeding_freq not in added_legend:
            label = f'Feeding Freq: {feeding_freq}/day'
            added_legend.add(feeding_freq)
        else:
            label = None  # Avoid duplicate legend entries

        plt.plot(group_sorted.index, group_sorted[column_name], '--', 
                 label=label, color=colors[feeding_freq])

    plt.xlabel('Time (Hours since 12 AM)')
    plt.ylabel(column_name)
    plt.title(f'{column_name} vs Time by Feeding Frequency')

    # Only create a legend for the labels we've defined (which excludes dates now)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


# In[ ]:


def plot_3D_time_series(df_graphing, column_name1, column_name2):
    # Plotting setup for 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Define colors for the different feeding frequencies
    colors = {1: 'blue', 2: 'green'}

    # Track the feeding frequencies that have been added to the legend
    added_legend = set()

    for (feeding_freq, date), group in df_graphing.groupby(['Feeding_Freq', 'Date']):
        group_sorted = group.sort_index()  # Sort the group by the index (TimeNumeric)

        # Check if this feeding frequency has already been added to the legend
        if feeding_freq not in added_legend:
            label = f'Feeding Freq: {feeding_freq}/day'
            added_legend.add(feeding_freq)
        else:
            label = None  # Avoid duplicate legend entries

        # Plot in 3D. X, Y, and Z coordinates correspond to index, 'Weighted_Position_x', and 'Weighted_Position_y' respectively
        ax.plot3D(group_sorted.index, group_sorted[column_name1], group_sorted[column_name2], 
                  '--', label=label, color=colors[feeding_freq],  alpha=0.7)

    # Set labels for the 3 axes
    ax.set_xlabel('Time (Hours since 12 AM)')
    ax.set_ylabel(column_name1)
    ax.set_zlabel(column_name2)
    plt.title(f'3D Plot of {column_name1} & {column_name2} vs Time by Feeding Frequency')

    # Only create a legend for the labels we've defined (which excludes dates now)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


# In[ ]:


def plot_1_stdev_band(df_graphing, column_name):
    plt.figure(figsize=(12, 8))

    # Define colors for the different feeding frequencies
    colors = {1: 'blue', 2: 'green'}  # Base colors for lines
    band_colors = {1: 'black', 2: 'orange'}  # Colors for the shaded bands

    added_legend = set()

    # Calculate mean and std for each feeding frequency across all dates first
    for feeding_freq, group_ff in df_graphing.groupby('Feeding_Freq'):
        mean = group_ff.groupby('TimeNumeric')[column_name].mean()
        std = group_ff.groupby('TimeNumeric')[column_name].std()

        # Shade the mean ± 1 standard deviation band
        plt.fill_between(mean.index, mean - std, mean + std, color=band_colors[feeding_freq], alpha=0.8, label=f'1 stdev band {feeding_freq}/day')

    # Now plot the individual lines for each date within each feeding frequency
    for (feeding_freq, date), group in df_graphing.groupby(['Feeding_Freq', 'Date']):
        group_sorted = group.sort_index()  # Sort by TimeNumeric

        if feeding_freq not in added_legend:
            label = f'Feeding Freq: {feeding_freq}/day'
            added_legend.add(feeding_freq)
        else:
            label = None

        # Make the individual lines less opaque
        plt.plot(group_sorted.index, group_sorted[column_name], '--', label=label, 
                 color=colors[feeding_freq], alpha=0.1)

    if 2 in df_graphing['Feeding_Freq'].unique():
        if len(df_graphing['Feeding_Freq'].unique()) == 2:
            plt.axvline(x=8, color='r', linestyle='solid', label='Feeding')
            plt.axvline(x=20, color='purple', linestyle='solid', label='Feeding (2/day only)')
            graph_type = 'combined'
        else:
            plt.axvline(x=8, color='r', linestyle='solid', label='Feeding')
            plt.axvline(x=20, color='r', linestyle='solid')
            graph_type = 'Twice'            
    else:
        plt.axvline(x=8, color='r', linestyle='solid', label='Feeding')
        graph_type = 'Once'
        
    plt.xlabel('Time (Hours since 12 AM)')
    plt.ylabel(column_name)
    plt.title(f'{column_name} vs Time by Feeding Frequency')
#     plt.ylim(df_graphing[column_name].min(), 90) #for once-a-day's Avg_Dir since there is an outlier with huge value
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    #Save the graph
    plt.savefig(f'{column_name}_{graph_type}.png', bbox_inches='tight')
    
    plt.show()


# In[ ]:


def plot_3D_1_stdev_band(df_graphing, column_name1, column_name2):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {1: 'blue', 2: 'green'}  # Base colors for lines
    band_colors = {1: 'black', 2: 'orange'}  # Alternate approach for 3D

    added_legend = set()
    
#     # Commented because it is unclear where the intersection of the plane with the graph lines are
#     # Determine the ranges of column_name1 and column_name2 for the planes
#     y_min, y_max = df_graphing[column_name1].min(), df_graphing[column_name1].max()
#     z_min, z_max = df_graphing[column_name2].min(), df_graphing[column_name2].max()

#     # Define the grid for plotting planes
#     y = np.linspace(y_min, y_max, 10)
#     z = np.linspace(z_min, z_max, 10)
#     Y, Z = np.meshgrid(y, z)

#     # Planes at specific x values to indicate feeding
#     for x_plane in [8, 20]:
#         X = x_plane * np.ones_like(Y)  # Create an array filled with the x_plane value
#         # Plot the plane with a slight transparency (alpha < 1)
#         ax.plot_surface(X, Y, Z, alpha=0.3, color='red', edgecolor='none')

    # Calculate mean and std for each feeding frequency across all dates first
    for feeding_freq, group_ff in df_graphing.groupby('Feeding_Freq'):
        mean = group_ff.groupby('TimeNumeric')[column_name1].mean()
        std = group_ff.groupby('TimeNumeric')[column_name1].std()
        z = group_ff.groupby('TimeNumeric')[column_name2].mean()  # Assuming you want to add a Z dimension

        # Plot the mean line in 3D
        ax.plot(mean.index, mean, z, label=f'Mean Freq: {feeding_freq}/day', color=band_colors[feeding_freq])

        # Plot lines for mean ± std as an alternative to fill_between in 3D
        ax.plot(mean.index, mean - std, z, color=band_colors[feeding_freq], alpha=0.8)
        ax.plot(mean.index, mean + std, z, color=band_colors[feeding_freq], alpha=0.8)

    # Now plot the individual lines for each date within each feeding frequency
    for (feeding_freq, date), group in df_graphing.groupby(['Feeding_Freq', 'Date']):
        group_sorted = group.sort_index()
        z = group_sorted[column_name2]  # The new Z dimension

        if feeding_freq not in added_legend:
            label = f'Lines Freq: {feeding_freq}/day'
            added_legend.add(feeding_freq)
        else:
            label = None

        # Plot the individual lines in 3D
        ax.plot3D(group_sorted.index, group_sorted[column_name1], z, '--', label=label, 
                  color=colors[feeding_freq], alpha=0.1)

    ax.set_xlabel('Time (Hours since 12 AM)')
    ax.set_ylabel(column_name1)
    ax.set_zlabel(column_name2)
    plt.title(f'3D Plot of {column_name1} & {column_name2} vs Time by Feeding Frequency')

    # Adjust legend to not repeat labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    #Save the graph
    plt.savefig(f'3Dgraph_{column_name1}{column_name2}.png', bbox_inches='tight')

    plt.show()

