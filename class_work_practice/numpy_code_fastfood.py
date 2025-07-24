import numpy as np

address, latitude , longitude , name, = np.genfromtxt('FastFoodRestaurants.csv',
                                                                delimiter=',', usecols=(0,4,5,6), unpack=True, dtype=None,skip_header=1)

print(address)
print(latitude)
print(longitude)
print(name)
# Zameen.com price  - statistics operations
print("FastFoodRestaurants latitude mean: " , np.mean(latitude))
print("FastFoodRestaurants latitude average: " , np.average(latitude))
print("FastFoodRestaurants latitude std: " , np.std(latitude))
print("FastFoodRestaurants latitude median: " , np.median(latitude))
print("FastFoodRestaurants latitude percentile - 25: " , np.percentile(latitude,25))
print("FastFoodRestaurants latitude  - 75: " , np.percentile(latitude,75))
print("FastFoodRestaurants latitude  - 3: " , np.percentile(latitude,3))
print("FastFoodRestaurants latitude min : " , np.min(latitude))
print("FastFoodRestaurants latitude max : " , np.max(latitude))

# Zameen.com price  - maths operations
print("FastFoodRestaurants latitude square: " , np.square(latitude))
print("FastFoodRestaurants latitude sqrt: " , np.sqrt(latitude))
print("FastFoodRestaurants latitude pow: " , np.power(latitude,latitude))
print("FastFoodRestaurants latitude abs: " , np.abs(latitude))

# Perform basic arithmetic operations
addition = latitude + longitude
subtraction = latitude - longitude
multiplication = latitude * longitude
division = latitude / longitude

print("FastFoodRestaurants latitude - longitude - Addition:", addition)
print("FastFoodRestaurants latitude - longitude - Subtraction:", subtraction)
print("FastFoodRestaurants latitude - longitude - Multiplication:", multiplication)
print("FastFoodRestaurants latitude - longitude - Division:", division)
#Trigonometric Functions
pricePie = (longitude/np.pi) +1

sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("FastFoodRestaurants latitude - div - pie  - Sine values:", sine_values)
print("FastFoodRestaurants latitude - div - pie Cosine values:", cosine_values)
print("FastFoodRestaurants latitude - div - pie Tangent values:", tangent_values)

print("FastFoodRestaurants latitude - div - pie  - Exponential values:", np.exp(pricePie))

