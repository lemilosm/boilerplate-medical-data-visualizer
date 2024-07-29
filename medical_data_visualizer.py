import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# 2
# df['overweight'] = None

def ow(Bmi):
    if Bmi>25:
        return 1
    else:
        return 0
    
bmi = round((df.weight / (df.height/100 )**2), 2 )
df['overweight']=bmi.apply(ow)
del(bmi)


# 3  Normalize data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, set the value to 0. 
# # If the value is more than 1, set the value to 1.
def normalize(digit):
    if digit == 1:
        return 0
    if digit > 1:
        return 1
df.cholesterol = df.cholesterol.apply(normalize)
df.gluc = df.gluc.apply(normalize)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 
    # 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. 
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    df_cat = df_cat.rename(columns={0: 'total'})



     # Draw the catplot with 'sns.catplot()'
#     # Get the figure for the output
#     fig = None#

    # Create a bar plot
    graph =sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')
    fig = graph.figure
    # fig = sns.catplot(data=df_cat, x='variable', y='total' , col='cardio', kind='bar', hue='value')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

#
#
# # Draw Heat Map
def draw_heat_map():

    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
                  ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14 Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # 15 Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
#see   https://seaborn.pydata.org/generated/seaborn.heatmap.html

    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig