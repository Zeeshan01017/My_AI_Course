mysaved_data = [.6,353,"Zeeshan","Allah dad","cricket",23,5.10,[87,70,90,95]]

print(mysaved_data)
print(type(mysaved_data))
for item in mysaved_data:
    print(item)
mysaved_data.append("123")
print(mysaved_data)
mysaved_data.insert(2,True)
print(mysaved_data)
mysaved_data.remove(7.6)
print(mysaved_data)
del mysaved_data[2]
print("after del: ",mysaved_data)
