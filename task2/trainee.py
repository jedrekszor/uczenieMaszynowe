import argparse
import fileinput

description = []
inputs = []
results = []

parser = argparse.ArgumentParser(description="description file")
parser.add_argument('-d', dest='filename', help='description file')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

with open(args.filename) as f:
    dim, pw = map(int, f.readline().split())
    lines = f.readlines()

for line in lines:
    description.append(list(map(float, line.split())))

for line in fileinput.input(files=args.files):
    inputs.append(list(map(float, line.split())))
fileinput.close()

for ex in inputs:
    res = 0
    for row in description:
        multiplier = row[pw]
        coef = 1
        for col in range(pw - 1, -1, -1):
            if row[col] == 0:
                continue
            coef *= ex[int(row[col] - 1)]
        res += coef * multiplier
    print(res)
