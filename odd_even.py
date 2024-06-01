def check_odd_even(number):
    return "Even" if number % 2 == 0 else "Odd"

try:
    number = int(input("Enter a number: "))
    print(f"The number {number} is {check_odd_even(number)}.")
except ValueError:
    print("Please enter a valid integer.")
