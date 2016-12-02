import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]
f = open(filename)
line = [float(l[:-1]) for l in f]
plt.plot(line)
plt.savefig('log.jpg')
