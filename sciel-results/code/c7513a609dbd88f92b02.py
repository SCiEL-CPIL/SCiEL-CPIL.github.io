import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    def gen_batcher(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = gen_batcher(half)
        right = gen_batcher(half)
        right_shifted = [(i + half, j + half) for i, j in right]
        merge_comps = odd_even_merge(half)
        return left + right_shifted + merge_comps

    def odd_even_merge(m):
        if m == 1:
            return [(0, 1)]
        
        comps = []
        for i in range(m):
            comps.append((i, m + i))
        
        half_m = m // 2
        even_merge = odd_even_merge(half_m)
        odd_merge = odd_even_merge(half_m)
        
        for i, j in even_merge:
            comps.append((2*i, 2*j))
        for i, j in odd_merge:
            comps.append((2*i+1, 2*j+1))
            
        return comps

    current_net = gen_batcher(16)
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)
    
    def verify_full(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)
    
    if not verify_full(current_net):
        # Fallback to initial
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = current_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    random.seed(42)
    
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break
        
        improved = False
        # Try to delete 1
        if net_len > 0:
            idx = random.randint(0, net_len - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify_full(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                improved = True
        
        if not improved:
            # Try to delete 2
            if net_len > 1:
                i1 = random.randint(0, net_len - 1)
                i2 = random.randint(0, net_len - 1)
                if i1 != i2:
                    if i1 > i2: i1, i2 = i2, i1
                    candidate = current_net[:i1] + current_net[i1+1:i2] + current_net[i2+1:]
                    if verify_full(candidate):
                        current_net = candidate
                        if len(current_net) < best_len:
                            best_net = current_net
                            best_len = len(current_net)
                        improved = True
        
        if not improved:
             # Try swap
             if net_len > 1:
                 i1 = random.randint(0, net_len - 1)
                 i2 = random.randint(0, net_len - 1)
                 if i1 != i2:
                     candidate = current_net.copy()
                     candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                     if verify_full(candidate):
                         current_net = candidate

    return best_net
# EVOLVE-BLOCK-END
