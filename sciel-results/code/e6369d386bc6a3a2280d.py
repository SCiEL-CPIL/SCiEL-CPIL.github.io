import time
import numpy as np

N = 16

def _merge(k):
    if k == 1:
        return [(0, 1)]
    k_half = k // 2
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    return even_comps + odd_comps + final_comps

def _batcher(n):
    if n == 1:
        return []
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    return left + right + merge

def _verify_network(comparators, inputs):
    state = inputs.copy()
    for i, j in comparators:
        swap_mask = (state[:, i] == 1) & (state[:, j] == 0)
        state[swap_mask, i] = 0
        state[swap_mask, j] = 1
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to be safe
    
    network = _batcher(N)
    
    # Precompute all inputs
    arange = np.arange(1 << N)
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (arange >> i) & 1
    
    # Quick check inputs: random sample
    np.random.seed(42)
    quick_indices = np.random.choice(1 << N, 2000, replace=False)
    quick_inputs = all_inputs[quick_indices]
    
    # Remove redundant comparators with multiple passes
    for pass_num in range(20):
        if time.time() > deadline:
            break
        changed = False
        if pass_num % 2 == 0:
            indices = range(len(network) - 1, -1, -1)
        else:
            indices = range(len(network))
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate, quick_inputs) and _verify_network(candidate, all_inputs):
                network = candidate
                changed = True
                break
        if not changed:
            break
    
    return network
