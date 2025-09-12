import numpy as np
import pandas as pd
from scipy import stats


file_path = "W2_Assignment/RealEstate-USA.csv"   
df = pd.read_csv(file_path)

# Extract required columns as NumPy arrays
brokered_by = df["brokered_by"].values
price = df["price"].values
acre_lot = df["acre_lot"].values
city = df["city"].values
house_size = df["house_size"].values

print("Arrays Loaded Successfully!")
print("Price Array Sample:", price[:5])
print("House Size Array Sample:", house_size[:5])

# 2. Descriptive Stats on Price

print("\n--- Price Statistics ---")
print("Mean:", np.mean(price))
print("Median:", np.median(price))
print("Mode:", stats.mode(price, keepdims=True)[0][0])
print("Standard Deviation:", np.std(price))
print("Variance:", np.var(price))
print("Minimum:", np.min(price))
print("Maximum:", np.max(price))

# 3. Descriptive Stats on House Size

print("\n--- House Size Statistics ---")
print("Mean:", np.mean(house_size))
print("Median:", np.median(house_size))
print("Mode:", stats.mode(house_size, keepdims=True)[0][0])
print("Standard Deviation:", np.std(house_size))
print("Variance:", np.var(house_size))
print("Minimum:", np.min(house_size))
print("Maximum:", np.max(house_size))

# 4. Arithmetic Operations

print("\n--- Arithmetic Operations ---")
print("Addition (+):", price[:5] + house_size[:5])
print("Addition (np.add):", np.add(price[:5], house_size[:5]))
print("Subtraction (-):", price[:5] - house_size[:5])
print("Subtraction (np.subtract):", np.subtract(price[:5], house_size[:5]))
print("Multiplication (*):", price[:5] * house_size[:5])
print("Multiplication (np.multiply):", np.multiply(price[:5], house_size[:5]))


# 5. 2D Array (Price & House Size)

array_2d = np.array([price, house_size])
print("\n--- 2D Array ---")
print(array_2d)

# 6. 3D Array (House Size, Price, Acre Lot)

array_3d = np.array([house_size, price, acre_lot])
print("\n--- 3D Array ---")
print(array_3d)

# 7. Iterate with np.nditer

print("\n--- Iterating with np.nditer (Price Array) ---")
for val in np.nditer(price[:10]):   
    print(val)

# 8. Iterate with np.ndenumerate

print("\n--- Iterating with np.ndenumerate (Price Array) ---")
for idx, val in np.ndenumerate(price[:10]):
    print(f"Index {idx} → Value {val}")

# 9. Array Properties

print("\n--- Array Properties of Price ---")
print("ndim:", price.ndim)
print("shape:", price.shape)
print("size:", price.size)
print("dtype:", price.dtype)
print("itemsize:", price.itemsize)
print("nbytes:", price.nbytes)
print("strides:", price.strides)

# 10. Slicing on 2D Array

print("\n--- Slicing (Q10) ---")
print(array_2d[0:3, 1:4])

# 11. Slicing on 2D Array

print("\n--- Slicing (Q11) ---")
print(array_2d[1:8, 2:5])

# 12. Geometric Functions

print("\n--- Geometric Operations ---")
print("sin:\n", np.sin(array_2d[:, :5]))
print("cos:\n", np.cos(array_2d[:, :5]))
print("tan:\n", np.tan(array_2d[:, :5]))
print("exp:\n", np.exp(array_2d[:, :5]))
print("log:\n", np.log(array_2d[:, 1:6]))
print("sqrt:\n", np.sqrt(array_2d[:, :5]))