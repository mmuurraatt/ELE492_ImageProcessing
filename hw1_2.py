def square(x):
    return x * x


def fahr_to_cent(fahr):
    return (fahr - 32) / 9.0 * 5


def cent_to_fahr(cent):
    result = cent / 5.0 * 9 + 32
    return result


def abs(x):
    if x < 0:
        return -x
    else:
        return x


def print_hello():
    print("Hello, world")


def print_fahr_to_cent(fahr):
    result = (fahr - 32) / 9.0 * 5
    print(result)


x = 42
square(3) + square(4)  # The result would have been 25 if it was printed
print(x)
boiling = fahr_to_cent(212)  # boiling becomes 100.0
cold = cent_to_fahr(-40)  # cold becomes -40.0
# print(result)      #This doesn't work because it is not defined above.
print(abs(-22))
print(print_fahr_to_cent(32))  # prints the centigrade value of 32 which is 0.0
# At the end it prints None because nothing is returned in print_fahr_to_cent() function
