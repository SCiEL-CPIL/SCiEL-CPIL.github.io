import numpy as np
import time

N = 17

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin

    # 1. Generate Seed Network: Odd-Even Transposition
    # Phase p: compare (i, i+1) for i in {p%2, p%2+2, ...}
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    
    # Pre-compute all 2^17 Boolean inputs
    # Shape: (2^17, 17)
    inputs = np.array([list(map(int, bin(i)[2:].zfill(N))) for i in range(1 << N)], dtype=np.uint8)
    
    def simulate(comps):
        state = inputs.copy()
        for i, j in comps:
            a = state[:, i]
            b = state[:, j]
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
        # Check if sorted: state[:, k] <= state[:, k+1] for all k
        # diff = np.diff(state, axis=1)
        # return np.all(diff >= 0)
        # Optimization: Check if any diff < 0
        return not np.any(np.diff(state, axis=1) < 0)

    # Verify seed (should be true)
    # if not simulate(comps):
    #     raise ValueError("Seed is invalid")

    # 2. Greedy Removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for k in range(len(comps)):
            if time.time() - start_time > deadline:
                break
            # Try removing comps[k]
            new_comps = comps[:k] + comps[k+1:]
            if simulate(new_comps):
                comps = new_comps
                improved = True
                break  # Restart scan to find new redundancies

    return comps
# EVOLVE-BLOCK-END
