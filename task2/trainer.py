import argparse
import fileinput
import polynomial
import sys

parser = argparse.ArgumentParser(description="description file")
parser.add_argument('-t', dest='train_set', help='train set')
parser.add_argument('-i', dest='data_in', help='data in')
parser.add_argument('-o', dest='data_out', help='data out')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

x = []
y = []
coef = []
inputs = []
dim, pw = 0, 0
learning_rate = 0.05
stop_condition = 0.00001

##### POLYNOMIAL INPUT #####
firstline = True
temp = []
for line in fileinput.input(files=args.files):
    if firstline:
        dim, pw = map(int, line.split())
        firstline = False
        continue
    temp.append(list(map(float, line.split())))
temp.sort(key=lambda col: col[0])
temp = list(reversed(temp))
for line in range(0, len(temp)):
    inputs.append(map(int, temp[line][:pw]))
    coef.append(temp[line][pw])

##### TRAIN DATA #####
with open(args.train_set) as f:
    lines = f.readlines()
for line in lines:
    line_list = list(map(float, line.split()))
    temp = []
    for i in range(0, dim):
        temp.append(line_list[i])
    x.append(temp)
    y.append(line_list[dim])

##### ITERATIONS #####
with open(args.data_in) as f:
    iterations = int(f.readline().split("=")[1])

######################################## TRAINING ########################
it = 0
for it in range(0, iterations):
    batch_error = [0] * (len(coef))
    for case in range(0, len(x)):
        base = y[case] - polynomial.calculate_polynomial(x[case], coef, inputs)

        for d in range(0, len(batch_error)):
            temp_batch_error = base
            if inputs[d][0] != 0:
                temp_batch_error *= x[case][inputs[d][0] - 1]
            batch_error[d] += temp_batch_error
    # gradient
    for deg in range(0, len(batch_error)):
        batch_error[deg] *= -2.0 / len(x)

    # adjusting coefficients
    new_coef = []
    for c in range(0, len(coef)):
        new_coef.append(coef[c] - learning_rate * batch_error[c])
    coef = new_coef

    # break condition
    can_stop = True
    for stop in batch_error:
        if abs(stop) > stop_condition:
            can_stop = False
    if can_stop:
        break

##### SAVE #####
with open(args.data_out, "w") as f:
    f.write("iterations=" + str(it + 1) + "\n")


sys.stdout.write(str(dim) + " " + str(pw) + "\n")
for line in range(0, len(coef)):
    for column in range(0, pw):
        sys.stdout.write(str(inputs[line][column]) + " ")
    sys.stdout.write(str(coef[line]) + "\n")
