import argparse
import random
import fileinput
import tools

parser = argparse.ArgumentParser(description="description file")
parser.add_argument('-t', dest='set', help='train set')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

degree = 1
m = 5
iterations = 1000
learning_rate = 0.5

# import set
x, y = tools.read_set(args)

# scale sets
x = tools.scale_x(x)
y = tools.scale_y(y)
dims = len(x[0])

final_coefs = []
means = []

while True:
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

                for row in range(0, len(coefs) - 1):
                    for col in range(0, len(coefs[row])):
                        temp_batch_error = base
                        for pos in range(1, col + 2):
                            temp_batch_error *= train_x[case][row]
                        batch_error[row][col] += temp_batch_error
                batch_error[-1][0] += base

            for row in range(0, len(batch_error)):
                for column in range(0, len(batch_error[row])):
                    batch_error[row][column] *= -2.0 / len(train_x)
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
    if len(means) != 0 and mean > means[-1]:
        break
    final_coefs.append(final_coef)
    means.append(mean/m)
    degree += 1
# print(means[-1])
# print(final_coefs[-1])
# print(degree - 1)

# test for in.txt
test_x = []
for line in fileinput.input(files=args.files):
    test_x.append(list(map(float, line.split())))

# save to out.txt
for test_case in test_x:
    print(tools.calculate_polynomial(test_case, final_coefs[-1]))
