import math


def read_set(args):
    x = []
    y = []
    with open(args.set) as f:
        lines = f.readlines()
    for line in lines:
        inp = list(map(float, line.split()))
        x.append(inp[:-1])
        y.append(inp[-1])
    return x, y


def scale_x(x):
    x_min = math.inf
    x_max = -math.inf
    for row in x:
        row_min = min(row)
        row_max = max(row)
        if row_min < x_min:
            x_min = row_min
        if row_max > x_max:
            x_max = row_max
    for row in x:
        for inp in range(0, len(row)):
            row[inp] = 2.0 * ((row[inp] - x_min) / (x_max - x_min)) - 1.0
    return x


def scale_y(y):
    y_min = min(y)
    y_max = max(y)
    for inp in range(0, len(y)):
        y[inp] = 2.0 * ((y[inp] - y_min) / (y_max - y_min)) - 1.0
    return y


def calculate_polynomial(x, coefs):
    result = 0
    for dim in range(0, len(coefs) - 1):
        for degree in range(0, len(coefs[dim])):
            argument = 1
            for pos in range(1, degree + 2):
                argument *= x[dim]
            result += argument * coefs[dim][degree]
    result += coefs[-1][0]
    return result
