import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

address, latitude , longitude , name, = np.genfromtxt("My_Projects_Work/US Food Restaurant DataSet/FastFoodRestaurants.csv",
                                                                delimiter=',',
                                                                usecols=(0, 4, 5, 6),
                                                                unpack=True,
                                                                dtype=('U100', 'f8', 'f8', 'U100'),   # force correct types
                                                                encoding='utf-8',
                                                                skip_header=1,
                                                                invalid_raise=False)

print(address)
print(latitude)
print(longitude)
print(name)

cleaned_longitude = longitude[~np.isnan(longitude)]
# Zameen.com price  - statistics operations
print("US Food Restaurant DataSet longitude mean:", np.nanmean(longitude))
print("US Food Restaurant DataSet longitude average:", np.nanmean(longitude))  # same as mean
print("US Food Restaurant DataSet longitude std:", np.nanstd(longitude))
print("US Food Restaurant DataSet longitude median: ",np.nanmedian(longitude))
print("US Food Restaurant DataSet longitude percentile - 25:", np.percentile(cleaned_longitude, 25))
print("US Food Restaurant DataSet longitude percentile - 75:", np.percentile(cleaned_longitude, 75))
print("US Food Restaurant DataSet longitude percentile - 3:", np.percentile(cleaned_longitude, 3))
print("US Food Restaurant DataSet longitude max:", np.nanmax(longitude))
print("US Food Restaurant DataSet longitude min:", np.nanmin(longitude))

# Zameen.com price  - maths operations
print("US Food Restaurant DataSet longitude square: ",np.square(longitude))
print("US Food Restaurant DataSet longitude sqrt: ",np.sqrt(longitude))
print("US Food Restaurant DataSet longitude pow: " , np.power(longitude,longitude))
print("US Food Restaurant DataSet longitude abs: " , np.abs(longitude))

# Perform basic arithmetic operations
addition = latitude + longitude
subtraction = latitude - longitude
multiplication = latitude * longitude
division = latitude / longitude

print("US Food Restaurant DataSet Long - lat - Addition:", addition)
print("US Food Restaurant DataSet Long - lat - Subtraction:", subtraction)
print("US Food Restaurant DataSet Long - lat - Multiplication:", multiplication)
print("US Food Restaurant DataSet Long - lat - Division:", division)

#Trigonometric Functions

pricePie = (longitude/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("US Food Restaurant DataSet longitude - div - pie  - Sine values:", sine_values)
print("US Food Restaurant DataSet longitude - div - pie Cosine values:", cosine_values)
print("US Food Restaurant DataSet longitude - div - pie Tangent values:", tangent_values)
print("US Food Restaurant DataSet longitude - div - pie  - Exponential values:", np.exp(pricePie))

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("US Food Restaurant DataSet longitude - div - pie  - Natural logarithm values:", log_array)
print("US Food Restaurant DataSet longitude - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("US Food Restaurant DataSet longitude - div - pie   - Hyperbolic Sine values:", sinh_values)

#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("US Food Restaurant DataSet longitude - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("US Food Restaurant DataSet longitude - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("US Food Restaurant DataSet longitude - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("US Food Restaurant DataSet longitude - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Zameen.com Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([latitude, longitude])

print ("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("US Food Restaurant DataSetm Long Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2LongLat):
    print(elem)


# EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)

"""# for loop
rows = np.shape(D2LongLat[0])[0]
cols = np.shape(D2LongLat[1])[0]
for i in range(0, (rows + 1)):
    for j in range(0, (cols + 1)):
        print (D2LongLat[i,j])
"""


# Reshape 2 x 149 into 1 x 298 using -1 to auto-infer columns
D2LongLat1TO298 = np.reshape(D2LongLat, (1, -1))
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : " , D2LongLat1TO298)
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : Size " , D2LongLat1TO298.size)
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : shape " , D2LongLat1TO298.shape)
print("US Food Restaurant DataSet Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)


print()