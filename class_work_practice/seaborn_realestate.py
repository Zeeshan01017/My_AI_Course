import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("class_work_practice/RealEstate-USA.csv",delimiter=",")

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)
print(df)
sns.set(style="whitegrid")
g=sns.displot(data=dffilter, x="house_size" , y="price" ,  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=house_size , y=price , hue=prev_sold_date,  kind='hist'  )"  )

# Display the plot
g.figure.show()
wait = input("wait...")

g=sns.lineplot(data=dffilter, x="house_size" , y="price"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=house_size , y=price  )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")