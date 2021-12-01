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


def calculate_polynomial(input_x, coeffs):
    y_pred = []
    for arg_num in range(0, len(input_x)):
        f = 0.0
        for x in coeffs:
            res = 1.0
            for y in range(0, len(x) - 1):
                if x[y] != 0:
                    res = res * input_x[arg_num][int(x[y]) - 1]
            res = res * float(x[len(x) - 1])
            f = f + res
        y_pred.append(f)
    return y_pred


def initialize_coefs(k, factors, n=1, depth=0):
    res = []
    if depth == k:
        res.append(factors)
        res[-1].append(1.0)
    else:
        for i in range(0, n + 1):
            new_factors = factors + [i]
            res.extend(initialize_coefs(k, new_factors, i, depth + 1))
    return res
