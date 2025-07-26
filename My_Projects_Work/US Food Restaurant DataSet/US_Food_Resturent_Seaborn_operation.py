# US_Food_Resturent_Dataset seaborn operation
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv('My_Projects_Work/US Food Restaurant DataSet/FastFoodRestaurants.csv',delimiter=",", index_col='keys')

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)

sns.set(style="whitegrid")

g=sns.displot(data=dffilter, x="name" , y="longitude" , hue="country",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=name , y=longitude , hue=country,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()
"""Plot univariate or bivariate histograms to show distributions of datasets.
A histogram is a classic visualization tool that represents the distribution
 of one or more variables by counting the number of observations that fall within discrete bins."""

g = sns.histplot(data=dffilter, x='name', y='longitude', hue='country', multiple="stack")
g.figure.suptitle("sns.histplot(data=dffilter, x='name', y='longitude', hue='country', multiple=stack)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()

# Use Seaborn to create a plot
g = sns.scatterplot(x='name', y='latitude', data=dffilter)
g.figure.suptitle("sns.scatterplot(x='name', y='latitude', data=dffilter)"  )
g.figure.show()
read = input("Wait for me....")

#line plot
g=sns.lineplot(data=dffilter, x="name" , y="longitude"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=name , y=lonitude  )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")


# barplot
g=sns.barplot(data=dffilter, x="name", y="latitude", legend=False)
g.figure.suptitle("sns.barplot(data=dffilter, x=name, y=latitude, legend=False)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")


#catplot
g=sns.catplot(data=dffilter, x="name", y="longitude")
g.figure.suptitle("sns.catplot(data=df, x=name, y=longitude)"  )
# Display the plot
g.figure.show() 
read = input("Wait for me....")

#heatmap
glue = dffilter.pivot(columns="name", values="longitude")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = dffilter.pivot(columns=name, values=longitude)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")