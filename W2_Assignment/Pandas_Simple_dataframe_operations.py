import pandas as pd
import numpy as np

# Question 1: Create DataFrame from Dictionary

data1 = {'X':[78,85,96,80,86], 
         'Y':[84,94,89,83,86],
         'Z':[86,97,96,72,83]}
df1 = pd.DataFrame(data1)
print("\n--- Q1: DataFrame from Dictionary ---")
print(df1)

# Question 2: DataFrame with Specified Index Labels

exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 
             'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 
                'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a','b','c','d','e','f','g','h','i','j']
df2 = pd.DataFrame(exam_data, index=labels)

print("\n--- Q2: DataFrame with Index Labels ---")
print(df2)


# Q2.1: Basic Summary

print("\n--- Q2.1: Info ---")
print(df2.info())

# Q2.2: First 3 Rows

print("\n--- Q2.2: First 3 Rows ---")
print(df2.head(3))

# Q2.3: Select name + score columns

print("\n--- Q2.3: name & score ---")
print(df2[['name', 'score']])

# Q2.4: Specific rows + columns

print("\n--- Q2.4: name & score for rows 1,3,5,6 ---")
print(df2.loc[['b','d','f','g'], ['name','score']])

# Q2.5: Rows where attempts > 2

print("\n--- Q2.5: attempts > 2 ---")
print(df2[df2['attempts'] > 2])

# Q2.6: Count rows & columns

print("\n--- Q2.6: Shape ---")
print("Rows:", df2.shape[0], "Columns:", df2.shape[1])

# Q2.7: Score between 15 & 20

print("\n--- Q2.7: Score between 15 and 20 ---")
print(df2[df2['score'].between(15,20)])

# Q2.8: attempts < 2 & score > 15

print("\n--- Q2.8: attempts < 2 & score > 15 ---")
print(df2[(df2['attempts'] < 2) & (df2['score'] > 15)])

# Q2.9: Change score in row 'd' to 11.5

df2.loc['d','score'] = 11.5
print("\n--- Q2.9: Updated row d ---")
print(df2.loc['d'])

# Q2.10: Mean of scores

print("\n--- Q2.10: Mean score ---")
print(df2['score'].mean())

# Q2.11: Append new row k, then delete

df2.loc['k'] = ['Alex', 15, 2, 'yes']
print("\n--- Q2.11: After adding row k ---")
print(df2.tail())

df2 = df2.drop('k')
print("\n--- After deleting row k (back to original) ---")
print(df2.tail())

# Q2.12: Sort by name DESC, score ASC

print("\n--- Q2.12: Sorted ---")
print(df2.sort_values(by=['name','score'], ascending=[False, True]))

# Q2.13: Replace qualify yes/no with True/False

df2['qualify'] = df2['qualify'].map({'yes': True, 'no': False})
print("\n--- Q2.13: Qualify replaced ---")
print(df2)

# Q2.14: Change name James → Suresh

df2['name'] = df2['name'].replace('James','Suresh')
print("\n--- Q2.14: Name changed ---")
print(df2)

# Q2.15: Delete attempts column

df3 = df2.drop(columns=['attempts'])
print("\n--- Q2.15: After dropping attempts ---")
print(df3)

# Q2.16: Write to CSV with tab separator

df2.to_csv("exam_data.tsv", sep='\t')
print("\n--- Q2.16: DataFrame written to exam_data.tsv (tab separated) ---")
