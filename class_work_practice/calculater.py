# create a calculater 
def add(x,y):
    return x + y
def subtract(x,y):
    return x - y
def multiply(x,y):
    return x * y
def divide(x,y):
    if y != 0:
        return x / y
    else:
        return"Error! Division by zero "
print("Select operation: ")
print("1.Addition")
print("2.Subtraction")
print("3.Multiplication")
print("4.Division")

choice = input("Enter choice (1/2/3/4): ")

number1 = float(input("Enter a first number: "))
number2 = float(input("Enter a second number: "))

if choice == '1':
    print(f"The result is: {add(number1,number2)}")
elif choice == '2':
    print(f"The result is: {subtract(number1, number2)}")
elif choice == '3':
    print(f"The result is: {multiply(number1, number2)}")
elif choice == '4':
    print(f"The result is: {divide(number1, number2)}")
else:
    print("Invalid Input")

