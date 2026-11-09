import time
import numpy as np

N = 16
DEADLINE = time.time() + 85  # safety margin before 90s kill

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires."""
    if n <= 1:
        return []
    
    def generate(start, length, direction):
        if length <= 1:
            return []
        
        half = length // 2
        comps = []
        
        # Sort halves in opposite directions
        comps.extend(generate(start, half, 1))
        comps.extend(generate(start + half, half, -1))
        
        # Bitonic Merge
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively merge
        comps.extend(generate(start, half, direction))
        comps.extend(generate(start + half, half, direction))
        
        return comps
    
    return generate(0, n, 1)

def _verify_full(network):
    """Full verification using numpy over all 65536 states."""
    states = np.arange(65536, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        states = np.where(swap, states ^ ((1 << i) | (1 << j)), states)
    return bool(np.all(((states + 1) & states) == 0))

def construct():
    start = time.time()
    
    # Start with Bitonic Sort (80 comparators)
    net = _bitonic_sort_network(N)
    best = list(net)
    best_len = len(best)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback to odd-even transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_len = len(best)

    import random
    rng = random.Random(12345)
    
    # Greedy removal with multiple restarts
    improved = True
    # We allow multiple passes with different random orders
    for restart in range(10):
        if time.time() > DEADLINE - 10:
            break
        
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            order = list(range(len(best)))
            rng.shuffle(order)
            for idx in order:
                if time.time() > DEADLINE - 10:
                    break
                candidate = best[:idx] + best[idx+1:]
                if _verify_full(candidate):
                    best = candidate
                    best_len = len(best)
                    improved = True

    # Final full verification
    if _verify_full(best):
        return best
    else:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

