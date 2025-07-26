# US_Food_Resturent_Dataset
import pandas as pd

df = pd.read_csv('My_Projects_Work/US Food Restaurant DataSet/FastFoodRestaurants.csv',delimiter=",")

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

name = df['name']
print("access the Name column: df : ")
print(name)
print()

# access multiple columns
name_country = df[['name','country']]
print("access multiple columns: df : ")
print(name_country)
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
second_row4 = df.loc[df['name'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:1,'name']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['name','country']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df.loc[:1,'address':'name']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['name'] == 'Gateway Properties','address':'name']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

print("# Case 2 : using .loc with index_col - starts here")

# Case 2 : using .loc with index_col - starts here
# Second cycle - with index_col as property_id
# Why Second cycle - Note Index - , index_col='property_id'

df_index_col = pd.read_csv('My_Projects_Work/US Food Restaurant DataSet/FastFoodRestaurants.csv',delimiter=",", index_col='keys')


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
print(df_index_col.index)
# Second cycle - with index_col as property_id

#Selecting a single row using .loc
second_row = df_index_col.loc['us/oh/washingtoncourthouse/530clintonave/-791445730']
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df_index_col.loc[['us/ky/maysville/408marketsquaredr/1051460804',
                                'us/ny/massena/6098statehighway37/-1161002137']]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

second_row3 = df_index_col.loc['us/oh/washingtoncourthouse/530clintonave/-791445730':'us/ky/maysville/408marketsquaredr/1051460804']
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

#Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['name'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df_index_col.loc[:'us/oh/washingtoncourthouse/530clintonave/-791445730','name']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df_index_col.loc[:'us/ky/maysville/408marketsquaredr/1051460804',['name','country']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df_index_col.loc[:'us/ny/massena/6098statehighway37/-1161002137','address':'name']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['name'] == 'Gateway Properties','address':'name']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

