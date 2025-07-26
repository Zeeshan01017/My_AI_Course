# Us real estate dataset seaborn operation
# Here we create different plots like displot, histplot, scatterplot, lineplot, barplot, catplot, heatplot
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("My_Projects_Work/US Real Estate Dataset/RealEstate-USA.csv",delimiter=",", index_col='brokered_by')

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)

sns.set(style="darkgrid")

# disply displot
g=sns.displot(data=dffilter, x="city" , y="price" , hue="house_size",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=city , y=price , hue=house_size,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")

#histplot show
g = sns.histplot(data=dffilter, x='city', y='price', hue='house_size', multiple="stack")
g.figure.suptitle("sns.histplot(data=dffilter, x='city', y='price', hue='house_size', multiple=stack)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()
dffilter = dffilter.loc[:, ~dffilter.columns.duplicated()]

# Reset index to avoid reindexing errors
dffilter = dffilter.reset_index(drop=True)
# Use Seaborn to create a plot
g = sns.scatterplot(x='price', y='house_size', data=dffilter)
g.figure.suptitle("sns.scatterplot(x='price', y='house_size', data=dffilter)"  )
g.figure.show()
read = input("Wait for me....")

#line plot
g=sns.lineplot(data=dffilter, x="house_size" , y="price"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=house_size , y=price  )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")


# barplot
g=sns.barplot(data=dffilter, x="house_size", y="price", legend=False)
g.figure.suptitle("sns.barplot(data=dffilter, x=house_size, y=price, legend=False)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")


#catplot
g=sns.catplot(data=dffilter, x="house_size", y="price")
g.figure.suptitle("sns.catplot(data=df, x=house_size, y=price)"  )
# Display the plot
g.figure.show() 
read = input("Wait for me....")

#heatmap
glue = dffilter.pivot(columns="house_size", values="price")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = dffilter.pivot(columns=house_size, values=price)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")



