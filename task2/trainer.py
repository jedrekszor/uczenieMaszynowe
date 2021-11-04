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
learning_rate = 0.06
stop_condition = 0.00001

##### POLYNOMIAL INPUT #####
firstline = True
for line in fileinput.input(files=args.files):
    if firstline:
        dim, pw = map(int, line.split())
        firstline = False
        continue
    inputs.append(list(map(float, line.split()))[:pw])
    coef.append(list(map(float, line.split()))[pw])
for line in range(0, len(inputs)):
    for column in range(0, len(inputs[line])):
        inputs[line][column] = int(inputs[line][column])

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
            batch_error[d] += base
            if inputs[d][0] == 0:
                continue
            batch_error[d] *= x[case][inputs[d][0] - 1]
    for deg in range(0, len(batch_error)):
        batch_error[deg] *= -2.0 / len(x)

    ##### GRADIENT #####
    new_coef = []
    for c in range(0, len(coef)):
        new_coef.append(coef[c] - learning_rate * batch_error[c])
    coef = new_coef
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
