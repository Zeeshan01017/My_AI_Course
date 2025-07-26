# startup growth investment dataset numpy operation
import numpy as np

Industry, Investment_Amount_USD , Number_of_Investors, Country = np.genfromtxt('My_Projects_Work/Startup Growth Investment Dataset/startup_growth_investment_data.csv',
                                                                                 delimiter=',', 
                                                                                 usecols=(1,3,5,6), 
                                                                                 unpack=True, 
                                                                                 dtype=None,
                                                                                 skip_header=1,
                                                                                 invalid_raise=False)

print(Industry)
print(Investment_Amount_USD)
print(Number_of_Investors)
print(Country)

# Zameen.com price  - statistics operations
print("startup_growth_investment_data Investment_Amount_USD mean: " , np.mean(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD average: " , np.average(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD std: " , np.std(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD mod: " , np.median(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD percentile - 25: " , np.percentile(Investment_Amount_USD,25))
print("startup_growth_investment_data Investment_Amount_USD percentile  - 50: " , np.percentile(Investment_Amount_USD,50))
print("startup_growth_investment_data Investment_Amount_USD percentile  - 75: " , np.percentile(Investment_Amount_USD,75))
print("startup_growth_investment_data Investment_Amount_USD min : " , np.min(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD max : " , np.max(Investment_Amount_USD))

# Zameen.com price  - maths operations
print("startup_growth_investment_data Investment_Amount_USD square: " , np.square(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD sqrt: " , np.sqrt(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD power: " , np.power(Investment_Amount_USD,Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD abs: " , np.abs(Investment_Amount_USD))


# Perform basic arithmetic operations
addition = Investment_Amount_USD + Number_of_Investors
subtraction = Investment_Amount_USD - Number_of_Investors
multiplication = Investment_Amount_USD * Number_of_Investors
division = Investment_Amount_USD / Number_of_Investors

print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - Addition:", addition)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - Subtraction:", subtraction)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - Multiplication:", multiplication)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - Division:", division)

#Trigonometric Functions

pricePie = (Investment_Amount_USD/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - div - pie  - Sine values:", sine_values)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - div - pie Cosine values:", cosine_values)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - div - pie Tangent values:", tangent_values)
print("startup_growth_investment_data Investment_Amount_USD - Number_of_Investors - div - pie  - Exponential values:", np.exp(pricePie))

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("startup_growth_investment_data Investment_Amount_USD  - div - pie  - Natural logarithm values:", log_array)
print("startup_growth_investment_data Investment_Amount_USD  - div - pie  = Base-10 logarithm values:", log10_array)

# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("startup_growth_investment_data Investment_Amount_USD -  div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("startup_growth_investment_data Investment_Amount_USD  - div - pie   - Hyperbolic Cosine values:", cosh_values)

# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("startup_growth_investment_data Investment_Amount_USD - div - pie   -Hyperbolic Tangent values:", tanh_values)

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("startup_growth_investment_data Investment_Amount_USD - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("startup_growth_investment_data Investment_Amount_USD - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Zameen.com Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([Investment_Amount_USD,Number_of_Investors])

print ("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - data type" ,D2LongLat.dtype)

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)

# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2LongLat):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)

D2LongLat1TO298 = np.reshape(D2LongLat, (1, -1))
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : " , D2LongLat1TO298)
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : Size " , D2LongLat1TO298.size)
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors - 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : shape " , D2LongLat1TO298.shape)
print("startup_growth_investment_data Investment_Amount_USD Plus Number_of_Investors- 2 dimentional arrary - np.reshape(D2LongLat, (1, -1)) : ndim " , D2LongLat1TO298.ndim)


print()