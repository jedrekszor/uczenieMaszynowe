import argparse
import random
import fileinput
import tools

parser = argparse.ArgumentParser(description="description file")
parser.add_argument('-t', dest='set', help='train set')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

degree = 1
m = 10
iterations = 1000
learning_rate = 0.2

# import set
x, y = tools.read_set(args)

# scale sets
# scale x
min_xs, max_xs = [], []
for xs_index in range(0, len(x[0])):
    min_xs.append(min(x, key=lambda i: i[xs_index])[xs_index])
    max_xs.append(max(x, key=lambda i: i[xs_index])[xs_index])

for xs in x:
    for x_index in range(0, len(xs)):
        xs[x_index] = 2*(xs[x_index]-min_xs[x_index])/(max_xs[x_index]-min_xs[x_index])-1

# scale y
min_y = min(y)
max_y = max(y)
y = [2*(i-min_y)/(max_y-min_y)-1 for i in y]




# x = tools.scale_x(x)
# y = tools.scale_y(y)
dims = len(x[0])

final_coefs = []
means = []

for power in range(0, 6):
    mean = 0
    final_coef = []
    for fold in range(0, m):
        # initialize coefficients
        coefs = []
        for d in range(0, dims):
            coefs.append([])
            for deg in range(0, degree):
                coefs[d].append(2 * random.random() - 1)
        coefs.append([2 * random.random() - 1])

        # perform m-fold CV
        lower_bound = int(len(x) * (fold / m))
        upper_bound = int(len(x) * ((fold + 1) / m))
        train_x = x[:lower_bound] + x[upper_bound:]
        train_y = y[:lower_bound] + y[upper_bound:]
        validate_x = x[lower_bound:upper_bound]
        validate_y = y[lower_bound:upper_bound]

        # train model
        for i in range(0, iterations):
            batch_error = []
            for row in range(0, len(coefs)):
                batch_error.append([])
                for column in range(0, len(coefs[row])):
                    batch_error[row].append(0)
            for case in range(0, len(train_x)):
                base = train_y[case] - tools.calculate_polynomial(train_x[case], coefs)
                # print(base)
                # exit(0)
                for row in range(0, len(coefs) - 1):
                    for col in range(0, len(coefs[row])):
                        temp_batch_error = base
                        for pos in range(1, col + 2):
                            temp_batch_error *= train_x[case][row]
                        batch_error[row][col] += temp_batch_error
                batch_error[-1][0] += base

            for row in range(0, len(batch_error)):
                for column in range(0, len(batch_error[row])):
                    batch_error[row][column] *= -1.0 / len(train_x)
            # print(batch_error)

            new_coefs = []
            for row in range(0, len(coefs)):
                new_coefs.append([])
                for column in range(0, len(coefs[row])):
                    new_coefs[row].append(coefs[row][column] - learning_rate * batch_error[row][column])
            coefs = new_coefs

        final_coef = coefs

        # calculate mean of the one fold
        q = 0
        for case in range(0, len(validate_x)):
            base = validate_y[case] - tools.calculate_polynomial(validate_x[case], coefs)
            q += base * base
        q /= len(train_x)
        mean += q
    mean /= m
    means.append(mean/m)
    if means[-1] == min(means):
        final_coefs = final_coef
    # final_coefs.append(final_coef)
    degree += 1
# print(means[-1])
# print(final_coefs[-1])
# print(degree - 1)

# test for in.txt
test_x = []
for line in fileinput.input(files=args.files):
    test_x.append(list(map(float, line.split())))

test_x = [[2 * (xs[x_index] - min_xs[x_index]) / (max_xs[x_index] - min_xs[x_index]) - 1 for x_index in range(0, len(xs))] for xs in test_x]

# save to out.txt
outs = []
for test_case in test_x:
    outs.append(tools.calculate_polynomial(test_case, final_coefs))
outs = [(p+1)/2.0*(max_y-min_y)+min_y for p in outs]

for o in outs:
    print(o)
