import fileinput
import sys
sort = []

for line in fileinput.input():
    sort.append(int(line))
fileinput.close()

sort.sort()

for num in sort:
    sys.stdout.write(str(num) + '\n')
