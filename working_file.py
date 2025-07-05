# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# %%
#importing data

df_test = pd.read_csv(r'train.csv')

df_test.head()

# %%
print(df_test.shape)

# %%
df_test.info()

# %%
df_test.describe()

# %%
#separate columns by data types

num_cols = df_test.select_dtypes(include = ['int64', 'float64']).columns
cat_cols = df_test.select_dtypes(include = ['object' , 'bool']).columns

print(num_cols)
print(cat_cols)


# %%
#checking for missing values

empty_values = df_test.isna().sum()
empty_values_ratio = empty_values / len(df_test)*100
empty_df = pd.DataFrame({'Values': empty_values, '_%_': empty_values_ratio})
empty_df = empty_df[empty_df['Values'] > 0]
empty_df


figure, ax1 = plt.subplots()
ax1.bar(empty_df.index, empty_df['Values'], color = 'red', alpha = 0.2,  label = 'empty_values')
ax1.set_ylabel('count_empty_values', fontsize = 12)
plt.xticks(rotation = 90)

ax2 = ax1.twinx()
ax2.plot(empty_df['_%_'], color = 'orange', alpha = 0.8, marker = 'o', label = '%_missing_values')
plt.show()




# %%
#inspecting 0 values

for col in num_cols.drop('id'):
    zero_count = (df_test[col] == 0).sum()
    zero_ratio = zero_count / len(df_test) * 100
    print(f'\n{col}:{zero_count}, {zero_ratio:.1f} % of overall')

# %%
#detecting outliers

def check_outlier(data,columns):
   for column in columns: 
        q1 = np.quantile(data[column].dropna(), 0.25)
        q3 = np.quantile(data[column].dropna(), 0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr * 1.5
        upper_bound = q3 + iqr * 1.5

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        print(f'Outliers in {column}:')

        print(outliers[[column]])
        print('-' * 40)

check_outlier(df_test, num_cols)




# %%
#plotting outliers

plt.subplots(figsize = (16,5))
for i,cols in enumerate(num_cols):
    plt.subplot(3,3,i+1)
    sns.boxplot(df_test, x = cols)
    plt.title(f'{cols}')
plt.tight_layout()
plt.show()
    


# %%
#visualising the spread



for cols in num_cols:
    figure, ax = plt.subplots()
    sns.histplot(df_test, x = cols,color = 'skyblue', kde= True, bins = 30)
    plt.xlabel(cols, fontsize = 12)
    plt.ylabel(cols, fontsize = 12)
    plt.grid(axis = 'y', linestyle = '--')
    ax.set_title(f'distribution_of: {cols}')
    plt.tight_layout()

    plt.show()

    #printing the descriptive stats

    print(f'descriptive_stats_of: {cols}')
    print(df_test[cols].describe(), '\n', '--'*40)

# %%
#exploring the categorical data

for cols in cat_cols.drop('Personality'):
    ax,figure = plt.subplots()
    sns.countplot(df_test, x = cols, palette = 'Set2', edgecolor = 'black')
    plt.xlabel('Indicator')
    plt.title(f'distribution of {cols}')

    plt.show()
    
    print(f'ratio of each category in {cols}, \n')
    print(df_test[cols].value_counts(normalize = True). round(2), '\n' + '--' * 40)





# %%
#exploring the categorical data

#the below graphs show the balance of each attributes vs the target variable


for cols in cat_cols.drop('Personality'):
    ax, figure = plt.subplots(figsize = (6,4))
    sns.countplot(df_test, x = cols, order = df_test[cols].value_counts().index, hue = 'Personality', palette = 'Set2', edgecolor = 'black')
    plt.xlabel('Indicator')
    plt.title(f'{cols}')
    plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()    



# %%
#exploring the strength of the correllation between the numerical variables

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']
ax,figure = plt.subplots()
sns.heatmap(df_test[num_cols].corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation between numerical features')
plt.show()




# %% [markdown]
# ##Analysis write-up
# 
# Overall patterns:
# 
# 1. Data set contains 18.5k rows. On average each fiels is missing 5% of values with the Stage_Fear having to the most of the missing values in the field(~10%)
# 
# 2. The data set has been evaluated for 0 values. Each field is within the acceptable range of <5%. Time_Spent_alone has got the highest amount ~17%, which could be explained as some people may not be spending any time alone during the day.
# 
# 3. Outliers check has been done. Time_spent alone is the only column that contains the outliers which contribute ~9% of all values.
# 
# 4. Skewness has been evaluated. Time_Spent_alone has got the skewness to the left and going_outside is slightly skewed to the right.
# 
# 5. Stage_fear and Drained_after_socialising are both skewed to "Yes" ~75% to 25% "No"
# 
# 6. Looking at the categorical features from the target perspective it is noticed that both extraverts and introverts are negatively correlated.
# 
# 7. Finally, the correlation between the num_variables has been explored and it could be observed that time_spent_alone is negatively correlated with the rest of the variables. While the rest are positively correlated with each other.
# 
# 
# Attribute findings:
# 
# 1. Time_Spend_alone - surveyed people spend 3.13 alone on average. IQR - 3 hours and 75% 4 hours. It indicates that there is a small minority of the surveyed who spend most of their day alone
# 
# 2. Social_Event_Attendance - 5hours on average. IQR - 5hours and 75% 8 hours which indicates the the majority dedicated about 5 hours to the social events.
# 
# 3. Going_outside - Avrg time outside - 4 hours. IQR - 3 hours and 75% - 6 hours. the following series are balanced and distributed closely around the mean.
# 
# 4. Friends_circle_size - avrg number of close friends - 8 people. IQR - 7 people. Full range 0-15 people, but it could be seen that the majority of observations lie in the IQR
# 
# 5. Post_Frequency - range - 0 to 10 posts. avrg posts -5 posts, IQR - 4 posts and 75% is within 7 posts.
# 
# 6. Stage_Fear -  as noted above it has been observed that 75% of respondents dont have a fear of stage.
# 
# 7. Drain_after_socialising - in like manner as stage_fear 75% of respondents dont experience fatigue after socialising.
# 


