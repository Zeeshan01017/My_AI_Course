# using numpy in startup growth investment dataset
import numpy as np

Industry, Investment_Amount_USD , Number_of_Investors, Country = np.genfromtxt('startup_growth_investment_data.csv', delimiter=',', usecols=(1,3,5,6), unpack=True, dtype=None,skip_header=1)

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
print("startup_growth_investment_data Investment_Amount_USD percentile  - 75: " , np.percentile(Investment_Amount_USD,75))
print("startup_growth_investment_data Investment_Amount_USD percentile  - 3: " , np.percentile(Investment_Amount_USD,3))
print("startup_growth_investment_data Investment_Amount_USD min : " , np.min(Investment_Amount_USD))
print("startup_growth_investment_data Investment_Amount_USD max : " , np.max(Investment_Amount_USD))

