from random import *

TIME_SPAN = 0.1


def generator():
    """Generates a csv containing 2 colums.
    The first one contains the time when a product has been sold.
    This corresponds with the 'x' variable, or independant variable
    The second one registers how much of te product has been sold.
    It corresponds to thw 'y' or dependant variable.
    """
    df = open("data2.csv", "a")
    sell = 0.0
    time = 9.5
    while time <= 13.5:
        sell += random()*2.5
        s = str(str(time) + "," + str(sell))
        df.write(s + "\n")
        time += TIME_SPAN
    while time <= 15.5:
        sell -= random()
        s = str(str(time) + "," + str(sell))
        df.write(s + "\n")
        time += TIME_SPAN
    while time <= 19:
        sell += random()*1.5
        s = str(str(time) + "," + str(sell))
        df.write(s + "\n")
        time += TIME_SPAN
    while time <= 21.5:
        sell -= random()*1.5
        s = str(str(time) + "," + str(sell))
        df.write(s + "\n")
        time += TIME_SPAN
    df.close()


def main():
    for i in range(1, 10):
        generator()

if __name__ == "__main__":
    main()
