# us real estate dataset pandas operation
import pandas as pd

df = pd.read_csv("My_Projects_Work/US Real Estate Dataset/RealEstate-USA.csv",delimiter=",")

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

city = df['city']
print("access the Name column: df : ")
print(city)
print()

# access multiple columns
city_price = df[['city','price']]
print("access multiple columns: df : ")
print(city_price)
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
second_row4 = df.loc[df['city'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:1,'city']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['city','price']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df.loc[:1,'city':'house_size']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['city'] == 'Gateway Properties','price':'city']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

print("# Case 2 : using .loc with index_col - starts here")

# Case 2 : using .loc with index_col - starts here
# Second cycle - with index_col as property_id
# Why Second cycle - Note Index - , index_col='property_id'

df_index_col = pd.read_csv('My_Projects_Work/US Real Estate Dataset/RealEstate-USA.csv',delimiter=",", index_col='brokered_by')


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
print(df_index_col.index)
# Second cycle - with index_col as property_id

#Selecting a single row using .loc
second_row = df_index_col.loc[31239]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df_index_col.loc[[31239, 1205]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

df_index_col = df_index_col[~df_index_col.index.duplicated()]
second_row3 = df_index_col.loc[62210:103378]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()

#Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['city'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df_index_col.loc[:92147,'city']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df_index_col.loc[:34632,['price','city']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df_index_col.loc[:31239,'price':'city']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['price'] == 'Gateway Properties','city':'price']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

