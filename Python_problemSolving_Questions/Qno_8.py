# calculate profit or loss
cost_price = int(input("Enter cost price: "))
selling_price = int(input("Enter selling price: "))
Amount = selling_price - cost_price
if cost_price < selling_price:
    print("Profit: ",Amount)
elif cost_price > selling_price:
    print("Loss: ",Amount)
else:
    print("NO profit - No Loss")