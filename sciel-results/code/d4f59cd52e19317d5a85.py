import time
import random
import numpy as np

N = 18
NUM_WORDS = (1 << N) // 64
TOTAL_INPUTS = 1 << N
indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
weights = 1 << np.arange(64, dtype=np.uint64)

# Pre-compute initial wires
initial_wires = [None] * N
for i in range(N):
    bits_i = (indices >> i) & 1
    bits_i_64 = bits_i.astype(np.uint64)
    packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
    initial_wires[i] = packed

def verify_network(net):
    # ... fast numpy check ...
    pass

def construct():
    # ... search logic ...
    pass
