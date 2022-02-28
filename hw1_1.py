fahr_Values = [-40, 0, 32, 68, 98.6, 212]
print("")
print("Fahrenheit       Centigrade")
for i in fahr_Values:
    print(str(i) + "                " + str((i-32)*5/9))
