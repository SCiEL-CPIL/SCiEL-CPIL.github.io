import time
import random

N = 16

# EVOLVE-BLOCK-START

def _apply_comparator(state, i, j):
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi == 0 and bj == 1:
        state |= (1 << i)
        state &= ~(1 << j)
    return state

def _is_sorted(state):
    if state == 0 or state == (1 << N) - 1:
        return True
    low = (state & -state).bit_length() - 1
    mask = ((1 << N) - 1) ^ ((1 << low) - 1)
    return state == mask

def _verify_fast(network):
    for mask in range(1 << N):
        state = mask
        for i, j in network:
            state = _apply_comparator(state, i, j)
            if _is_sorted(state):
                break
        else:
            if not _is_sorted(state):
                return False
    return True

def _build_oet(n):
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _remove_redundant_comparators(network, deadline):
    if not network:
        return network
    if not _verify_fast(network):
        return network

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            candidate = network[:idx] + network[idx+1:]
            if _verify_fast(candidate):
                network = candidate
                changed = True
                break
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    network = _build_oet(N)
    
    # Prune
    network = _remove_redundant_comparators(network, deadline)
    
    # Local Search
    best_network = network[:]
    best_len = len(network)
    
    while time.time() < deadline:
        # Try removing a random comparator
        if len(best_network) > 0 and random.random() < 0.7:
            idx = random.randint(0, len(best_network) - 1)
            candidate = best_network[:idx] + best_network[idx+1:]
            if _verify_fast(candidate):
                best_network = candidate
                best_len = len(candidate)
        else:
            # Try swapping
            if len(best_network) > 1:
                idx1 = random.randint(0, len(best_network) - 1)
                idx2 = random.randint(0, len(best_network) - 1)
                if idx1 != idx2:
                    candidate = best_network[:]
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    if _verify_fast(candidate):
                        best_network = candidate
    
    return best_network

# EVOLVE-BLOCK-END
