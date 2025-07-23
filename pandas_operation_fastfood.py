import pandas as pd

df = pd.read_csv('FastFoodRestaurants.csv',delimiter=",")

print(df)
print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )
#disply last three rows
print('Last three Rows:')
print(df.tail(3))
#disply first three rows
print('First Three Rows:')
print(df.head(3))
print()

print("Summary of Statistics of DataFrame using describe() method", df.describe())
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()

country = df['country']
print("access the Name column: df : ")
print(country)
print()

country_keys = df[['country','keys']]
print("access multiple columns: df : ")
print(country_keys)
print()

second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

second_row2 = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

second_row3 = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

second_row4 = df.loc[df['country'] == 'address']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

second_row5 = df.loc[:1,'country']
print("#Selecting a single column using .loc")
print(second_row5)
print()

second_row6 = df.loc[:1,['country','keys']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

second_row7 = df.loc[:1,'adress':'country']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

second_row8 = df.loc[df['country'] == 'keys','address':'country']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

#case 2
df_index_col = pd.read_csv('FastFoodRestaurants.csv',delimiter=",", index_col='address')

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

second_row = df_index_col.loc[38.06753]
print("#Selecting a single row using .loc")
print(second_row)
print()