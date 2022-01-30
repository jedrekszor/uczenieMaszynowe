import argparse
import fileinput
import tools

parser = argparse.ArgumentParser(description="description file")
parser.add_argument('-t', dest='set', help='train set')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

m = 10
iterations = 1200
learning_rate = 1.0

# import set
x, y = tools.read_set(args)

# scale x
min_xs, max_xs = [], []
for xs_index in range(0, len(x[0])):
    min_xs.append(min(x, key=lambda i: i[xs_index])[xs_index])
    max_xs.append(max(x, key=lambda i: i[xs_index])[xs_index])

for xs in x:
    for x_index in range(0, len(xs)):
        xs[x_index] = 2 * (xs[x_index] - min_xs[x_index]) / (max_xs[x_index] - min_xs[x_index]) - 1

# scale y
min_y = min(y)
max_y = max(y)
y = [2 * (i - min_y) / (max_y - min_y) - 1 for i in y]
dims = len(x[0])

best_deg = 0
means = []

for degree in range(1, 7):
    mean = 0
    final_coef = []
    for fold in range(0, m):

        # initialize coefficients
        coefs = tools.initialize_coefs(degree, [], dims)

        # performing m-fold CV
        lower_bound = int(len(x) * (fold / m))
        upper_bound = int(len(x) * ((fold + 1) / m))
        train_x = x[:lower_bound] + x[upper_bound:]
        train_y = y[:lower_bound] + y[upper_bound:]
        validate_x = x[lower_bound:upper_bound]
        validate_y = y[lower_bound:upper_bound]

        # training model
        for i in range(0, iterations):
            batch_error = []
            for row in coefs:
                batch_error.append([])

            # calculating error
            res = tools.calculate_polynomial(train_x, coefs)
            for case in range(0, len(train_x)):
                base = train_y[case] - res[case]
                batch_error[0].append(base)
                for row in range(1, len(coefs)):
                    result = 1.0
                    for c_info in coefs[row]:
                        if not isinstance(c_info, float) and c_info != 0:
                            result *= train_x[case][c_info - 1]
                    batch_error[row].append(result * base)

            # calculating gradient
            gradient = []
            for err in batch_error:
                gradient.append(-1.0 / len(train_x) * sum(err))

            # adjusting coefficients
            for c in range(0, len(gradient)):
                coeff = coefs[c][len(coefs[0]) - 1]
                coefs[c][len(coefs[0]) - 1] = coeff - learning_rate * gradient[c]

            # break condition
            if i > 0 and all(abs(g) < 0.0000001 for g in gradient):
                break

        final_coef = coefs

        # calculating mean of one fold
        q = 0
        res = tools.calculate_polynomial(validate_x, final_coef)
        for case in range(0, len(validate_x)):
            base = validate_y[case] - res[case]
            q += base * base
        q /= len(validate_x)
        mean += q
    # calculating mean of entire degree
    mean /= m
    means.append(mean)

    # checking for best degree
    if means[-1] == min(means):
        best_deg = degree

# after establishing the best degree, train again for the entire set
coefs = tools.initialize_coefs(best_deg, [], dims)
# training model
for i in range(0, iterations):
    batch_error = []
    for row in coefs:
        batch_error.append([])

    # calculating error
    res = tools.calculate_polynomial(x, coefs)
    for case in range(0, len(x)):
        base = y[case] - res[case]
        batch_error[0].append(base)
        for row in range(1, len(coefs)):
            result = 1.0
            for c_info in coefs[row]:
                if not isinstance(c_info, float) and c_info != 0:
                    result *= x[case][c_info - 1]
            batch_error[row].append(result * base)

    # calculating gradient
    gradient = []
    for err in batch_error:
        gradient.append(-1.0 / len(x) * sum(err))

    # adjusting coefficients
    for c in range(0, len(gradient)):
        coeff = coefs[c][len(coefs[0]) - 1]
        coefs[c][len(coefs[0]) - 1] = coeff - learning_rate * gradient[c]

    # break condition
    if i > 0 and all(abs(g) < 0.000001 for g in gradient):
        break

# reading in.txt
test_x = []
for line in fileinput.input(files=args.files):
    test_x.append(list(map(float, line.split())))

#scaling test_x
test_x = [
    [2 * (xs[x_index] - min_xs[x_index]) / (max_xs[x_index] - min_xs[x_index]) - 1 for x_index in range(0, len(xs))] for
    xs in test_x]

# saving to out.txt
outs = []
outputs = tools.calculate_polynomial(test_x, coefs)
# rescaling output
outputs = [(p + 1) / 2.0 * (max_y - min_y) + min_y for p in outputs]
for outp in outputs:
    print(outp)
