# take input and calculate total marks, average and percentage
a = int(input("Enter first subject marks: "))
b = int(input("Enter second subject marks: "))
c = int(input("Enter third subject marks: "))
d = int(input("Enter forth subject marks: "))
e = int(input("Enter fifth subject marks: "))

Maximum_Marks = 500
Total_Marks = a+b+c+d+e
Percentage = (Total_Marks/Maximum_Marks)*100
Average =(a+b+c+d+e)/5

print("Total Marks = ",Total_Marks)
print("Percentage of marks = ",Percentage,"%")
print("Average of marks = ",Average)



