import pandas as pd

url = "W2_Assignment/RealEstate-USA.csv"
df = pd.read_csv(url)

print("\n--- DataFrame ---")
print(df.head())

# 2. Info, dtypes, describe, shape

print("\n--- df.info() ---")
print(df.info())

print("\n--- dtypes ---")
print(df.dtypes)

print("\n--- describe() ---")
print(df.describe(include="all"))

print("\n--- shape ---")
print(df.shape)

# 3. to_string() experiments

print("\n--- to_string Example ---")
print(df.to_string(max_rows=5, max_cols=5, show_dimensions=True))

# 4. Top 7 rows

print("\n--- Top 7 rows ---")
print(df.head(7))

# 5. Bottom 9 rows

print("\n--- Bottom 9 rows ---")
print(df.tail(9))

# 6. Access city and street

print("\n--- City Column ---")
print(df["city"])

print("\n--- Street Column ---")
print(df["street"])

# 7. Access multiple columns

print("\n--- Street + City ---")
print(df[["street", "city"]])

# 8. Single row with loc

print("\n--- Row index 5 ---")
print(df.loc[5])

# 9. Multiple rows with loc

print("\n--- Rows 3,5,7 ---")
print(df.loc[[3, 5, 7]])

# 10. Slice rows with loc

print("\n--- Rows 3 to 9 ---")
print(df.loc[3:9])

# 11. Conditional selection (price > 100000)

print("\n--- Price > 100000 ---")
print(df.loc[df["price"] > 100000])

# 12. Conditional selection (city = Adjuntas)

print("\n--- City = Adjuntas ---")
print(df.loc[df["city"] == "Adjuntas"])

# 13. Conditional selection (city = Adjuntas & price < 180500)

print("\n--- Adjuntas & price < 180500 ---")
print(df.loc[(df["city"] == "Adjuntas") & (df["price"] < 180500)])

# 14. Select specific columns for row 7

print("\n--- Row 7 specific cols ---")
print(df.loc[7, ["city", "price", "street", "zip_code", "acre_lot"]])

# 15. Slice of columns with loc

print("\n--- Columns city to zip_code ---")
print(df.loc[:, "city":"zip_code"].head())

# 16. Combined row + column with loc

print("\n--- Adjuntas + city→zip_code ---")
print(df.loc[df["city"] == "Adjuntas", "city":"zip_code"])

# 17. Single row with iloc

print("\n--- 5th row ---")
print(df.iloc[5])

# 18. Multiple rows with iloc

print("\n--- 7th, 9th, 15th rows ---")
print(df.iloc[[7, 9, 15]])

# 19. Slice rows with iloc

print("\n--- 5th to 13th rows ---")
print(df.iloc[5:14])

# 20. Single column with iloc

print("\n--- 3rd column ---")
print(df.iloc[:, 2])

# 21. Multiple columns with iloc

print("\n--- 2nd, 4th, 7th columns ---")
print(df.iloc[:, [1, 3, 6]])

# 22. Slice of columns with iloc

print("\n--- 2nd to 5th columns ---")
print(df.iloc[:, 1:5])

# 23. Combined row+col with iloc

print("\n--- Rows [7,9,15] Cols [2,4] ---")
print(df.iloc[[7, 9, 15], [1, 3]])

# 24. Combined row+col slice with iloc

print("\n--- Rows 2 to 6, Cols 2 to 4 ---")
print(df.iloc[2:7, 1:4])

# 25. Add new row

new_row = {"city": "NewCity", "price": 123456, "street": "New Street", "zip_code": "99999"}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print("\n--- After Adding New Row ---")
print(df.tail())

# 26. Delete row with index 2

df = df.drop(index=2)
print("\n--- After Dropping Row 2 ---")
print(df.head())

# 27. Delete rows 4 to 7

df = df.drop(index=range(4, 8))
print("\n--- After Dropping Rows 4-7 ---")
print(df.head(10))

# 28. Delete house_size column

df1 = df.drop(columns=["house_size"])
print("\n--- After Dropping house_size ---")
print(df1.head())

# 29. Delete house_size + state columns

df2 = df.drop(columns=["house_size", "state"])
print("\n--- After Dropping house_size + state ---")
print(df2.head())

# 30. Rename column state to state_Changed

df3 = df.rename(columns={"state": "state_Changed"})
print("\n--- After Renaming state ---")
print(df3.head())

# 31. Rename row index 3 to 5

df4 = df.rename(index={3: 5})
print("\n--- After Renaming index 3 to 5 ---")
print(df4.head(10))

# 32. query()

print("\n--- Query price<127400 & city!=Adjuntas ---")
print(df.query("price < 127400 and city != 'Adjuntas'"))

# 33. Sort by price

print("\n--- Sorted by price ASC ---")
print(df.sort_values(by="price").head())

# 34. Group by city sum(price)

print("\n--- Group by city, sum(price) ---")
print(df.groupby("city")["price"].sum())

# 35. dropna()

print("\n--- After dropna() ---")
print(df.dropna().head())

# 36. fillna()

print("\n--- After fillna(0) ---")
print(df.fillna(0).head())
