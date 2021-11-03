import argparse
import fileinput
import polynomial

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
learning_rate = 0.01
stop_condition = 0.00001

##### POLYNOMIAL INPUT #####
firstline = True
for line in fileinput.input(files=args.files):
    if firstline:
        dim, pw = map(int, line.split())
        firstline = False
        continue
    inputs.append(list(map(float, line.split()))[:dim])
    coef.append(list(map(float, line.split()))[dim])
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
    line, iterations = f.readline().split("=")

######################################## TRAINING ########################
it = 0
for it in range(0, int(iterations)):
    batch_error = [0] * (len(coef))
    for case in range(0, len(x)):
        for d in range(0, len(batch_error)):
            batch_error[d] += y[case] - polynomial.calculate_polynomial(x[case], coef, inputs)
            for i in range(0, len(x[case])):
                if inputs[d][i] == 1:
                    batch_error[d] *= x[case][i]
    for deg in range(0, len(batch_error)):
        batch_error[deg] *= -2 / len(x)

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
    f.write("iterations=" + str(it))

print(str(dim) + " " + str(pw))
for line in range(0, len(coef)):
    print(str(*inputs[line]) + " " + str(coef[line]))
