"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    # Build initial wire values: wire[k] is a big int where bit i = value of wire k for input i
    wires_init = []
    for k in range(N):
        w = 0
        period = 1 << (k + 1)
        half = 1 << k
        for start in range(half, period, period):
            mask = ((1 << half) - 1) << start
            w |= mask
        wires_init.append(w)
    
    def verify(network):
        wires = wires_init[:]
        for (i, j) in network:
            wi, wj = wires[i], wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for k in range(N - 1):
            if wires[k] & ~wires[k + 1]:
                return False
        return True
    
    # Start from odd-even transposition
    best_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    if not verify(best_network):
        return best_network
    
    random.seed(42)
    current = best_network[:]
    best_len = len(current)
    
    # Aggressive removal: repeatedly try to remove random comparators
    for attempt in range(50000):
        if time.time() > deadline - 10:
            break
        if len(current) <= 40:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current[:idx] + current[idx + 1:]
        if verify(candidate):
            current = candidate
            if len(current) < best_len:
                best_len = len(current)
                best_network = current[:]
    
    # Now try to add comparators and then remove more (to escape local minima)
    for attempt in range(100000):
        if time.time() > deadline - 5:
            break
        
        r = random.random()
        if r < 0.6 and len(current) > 40:
            # Remove
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]
            delta = -1
        elif r < 0.8:
            # Add
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate = current + [(i, j)]
            delta = 1
        else:
            # Swap
            if len(current) < 2:
                continue
            i1, i2 = random.sample(range(len(current)), 2)
            candidate = current[:]
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            delta = 0
        
        if not verify(candidate):
            continue
        
        if delta < 0:
            current = candidate
            if len(current) < best_len:
                best_len = len(current)
                best_network = current[:]
        else:
            # Accept with some probability to escape local minima
            if random.random() < 0.3:
                current = candidate
    
    return best_network
# EVOLVE-BLOCK-END
