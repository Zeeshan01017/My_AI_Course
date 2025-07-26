# Startup growth investment dataset seaborn operation
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv('My_Projects_Work/Startup Growth Investment Dataset/startup_growth_investment_data.csv',
                delimiter=",",
                index_col='Industry')

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)

sns.set(style="dark")

g=sns.displot(data=dffilter, x="Industry" , y="Number of Investors" , hue="Country",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=Industry , y=Number of Investors , hue=Country,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")

g = sns.histplot(data=dffilter, x='Industry', y='Number of Investors', hue='Country', multiple="stack")
g.figure.suptitle("sns.histplot(data=dffilter, x='Industry', y='Number of Investors', hue='Country', multiple=stack)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")

# Use Seaborn to create a plot
g = sns.scatterplot(x='Industry', y='Number of Investors', data=dffilter)
g.figure.suptitle("sns.scatterplot(x='name', y='Number of Investors', data=dffilter)"  )
g.figure.show()
read = input("Wait for me....")

#line plot
g=sns.lineplot(data=dffilter, x="Industry" , y="Number of Investors"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=Industry , y=Number of Investors  )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")

# barplot
g=sns.barplot(data=dffilter, x="Industry", y="Number of Investors", legend=False)
g.figure.suptitle("sns.barplot(data=dffilter, x=Industry, y=Number of Investors, legend=False)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")


#catplot
g=sns.catplot(data=dffilter, x="Industry", y="Number of Investors")
g.figure.suptitle("sns.catplot(data=df, x=Industry, y=Number of Investors)"  )
# Display the plot
g.figure.show() 
read = input("Wait for me....")

