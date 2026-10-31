import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    # Single spike is optimal: ratio = N = 50
    h = [0.0] * N
    h[N // 2] = 1.0
    return h
# EVOLVE-BLOCK-END
