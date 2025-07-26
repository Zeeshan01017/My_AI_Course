# startup growth investment dataset with pandas operation
import pandas as pd

df = pd.read_csv("My_Projects_Work/Startup Growth Investment Dataset/startup_growth_investment_data.csv",delimiter=",")

print(df)

print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )

# display the first three rows
print('First five Rows:')
print(df.head(5))
print()

# display the last three rows
print('Last five Rows:')
print(df.tail(5))

#Summary of Statistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()

Industry = df['Industry']
print("access the Name column: df : ")
print(Industry)
print()

# access multiple columns

Industry_Number_of_Investors = df[['Industry','Number of Investors']]
print("access multiple columns: df : ")
print(Industry_Number_of_Investors)
print()

#Selecting a single row using .loc
second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

#Selecting a slice of rows using .loc
second_row3 = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

#Conditional selection of rows using .loc
second_row4 = df.loc[df['Industry'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:1,'Industry']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['Industry','Number of Investors']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df.loc[:1,'Industry':'Number of Investors']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['Industry'] == 'Gateway Properties','Industry':'Number of Investors']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

print("# Case 2 : using .loc with index_col - starts here")

# Case 2 : using .loc with index_col - starts here
# Second cycle - with index_col as property_id
# Why Second cycle - Note Index - , index_col='property_id'

df_index_col = pd.read_csv('My_Projects_Work/Startup Growth Investment Dataset/startup_growth_investment_data.csv',delimiter=",", index_col='Startup Name')


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
print(df_index_col.index)

#Selecting a single row using .loc
second_row = df_index_col.loc['Startup_4999']
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df_index_col.loc[['Startup_3', 'Startup_5']]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

df_index_col = df_index_col[~df_index_col.index.duplicated()]
second_row3 = df_index_col.loc['Startup_8':'Startup_10']
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

#Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['Industry'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df_index_col.loc[:'Startup_5','Industry']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df_index_col.loc[:'Startup_10',['Number of Investors','Industry']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df_index_col.loc[:'Startup_8','Number of Investors':'Industry']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['Number of Investors'] == 'Gateway Properties','Industry':'Number of Investors']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

