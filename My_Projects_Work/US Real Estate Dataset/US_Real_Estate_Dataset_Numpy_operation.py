# US real estate data set numoy operation
import numpy as np

brokered_by, price, city, house_size = np.genfromtxt("My_Projects_Work/US Real Estate Dataset/RealEstate-USA.csv",
                                                                                                        delimiter=",",
                                                                                                        usecols=(0,2,7,10),
                                                                                                        unpack=True,
                                                                                                        dtype=None,
                                                                                                        skip_header=1)
print(brokered_by)
print(price)
print(city)
print(house_size)

# Zameen.com price  - statistics operations
print("US Real Estate Price mean: " , np.mean(price))
print("US Real Estate Price average: " , np.average(price))
print("US Real Estate Price std: " , np.std(price))
print("US Real Estate Price mod: " , np.median(price))
print("US Real Estate Price percentile - 25: " , np.percentile(price,25))
print("US Real Estate Price percentile  - 50: " , np.percentile(price,50))
print("US Real Estate Price percentile  - 75: " , np.percentile(price,75))
print("US Real Estate Price min : " , np.min(price))
print("US Real Estate Price max : " , np.max(price))

# Zameen.com price  - maths operations
print("US Real Estate Price square: " , np.square(price))
print("US Real Estate Price sqrt: " , np.sqrt(price))
print("US Real Estate Price pow: " , np.power(price,price))
print("US Real Estate Price abs: " , np.abs(price))


# Perform basic arithmetic operations
addition = price + house_size
subtraction = price - house_size
multiplication = price * house_size
division = price / house_size

print("US Real Estate Price - house_size - Addition:", addition)
print("US Real Estate Price - house_size - Subtraction:", subtraction)
print("US Real Estate Price - house_size - Multiplication:", multiplication)
print("US Real Estate Price - house_size - Division:", division)

#Trigonometric Functions

pricePie = (price/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("US Real Estate Price - div - pie  - Sine values:", sine_values)
print("US Real Estate Price - div - pie Cosine values:", cosine_values)
print("US Real Estate Price - div - pie Tangent values:", tangent_values)
print("US Real Estate Price - div - pie  - Exponential values:", np.exp(pricePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("US Real Estate Price - div - pie  - Natural logarithm values:", log_array)
print("US Real Estate Price - div - pie  = Base-10 logarithm values:", log10_array)

# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("US Real Estate Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("US Real Estate Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("US Real Estate Price - div - pie   -Hyperbolic Tangent values:", tanh_values)

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("US Real Estate Price - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("US Real Estate Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Zameen.com Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([price,house_size])

print ("US Real Estate Price Plus house_size - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("US Real Estate Price Plus house_size - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("US Real Estate Price Plus house_size - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("US Real Estate Price Plus house_size - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("US Real Estate Price Plus house_size - 2 dimentional arrary - data type" ,D2LongLat.dtype)

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("US Real Estate Price Plus house_size - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("US Real Estate Price Plus house_size - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("US Real Estate Price Plus house_size - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("US Real Estate Price Plus house_size - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2LongLat):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)

D2LongLat1TO298 = np.reshape(D2LongLat, (1, -1))
print("US Real Estate Price Plus house_size - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : " , D2LongLat1TO298)
print("US Real Estate Price Plus house_size - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : Size " , D2LongLat1TO298.size)
print("US Real Estate Price Plus house_size - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)
print("US Real Estate Price Plus house_size - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : shape " , D2LongLat1TO298.shape)
print("US Real Estate Price Plus house_size - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)


print()