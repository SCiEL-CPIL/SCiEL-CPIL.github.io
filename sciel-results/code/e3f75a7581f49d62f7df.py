import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 145.0
    n = 16
    
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    core_mask = [0]*n
    for u, v in groetzsch_edges:
        core_mask[u] |= (1 << v)
        core_mask[v] |= (1 << u)
        
    best_count = 0
    best_mask = [0]*n
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initial state
    mask = [0]*n
    for u, v in groetzsch_edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    edge_count = len(groetzsch_edges)
    best_count = edge_count
    best_mask = mask[:]
    
    T = 5.0
    cooling = 0.999
    
    def elapsed_ok():
        return time.time() < deadline
        
    while elapsed_ok():
        # Choose move
        # 70% Add, 30% Remove (non-core)
        if random.random() < 0.7:
            u, v = random.choice(all_pairs)
            if not (mask[u] & (1 << v)):
                if not (mask[u] & mask[v]):
                    # Valid add
                    mask[u] |= (1 << v)
                    mask[v] |= (1 << u)
                    edge_count += 1
                    if edge_count > best_count:
                        best_count = edge_count
                        best_mask = mask[:]
                    # Accept improvement always
                else:
                    # Invalid, do nothing
                    pass
        else:
            # Remove non-core edge
            # Find a non-core edge to remove
            # We can pick a random pair, check if it's an edge and not core
            u, v = random.choice(all_pairs)
            if (mask[u] & (1 << v)) and not (core_mask[u] & (1 << v)):
                mask[u] &= ~(1 << v)
                mask[v] &= ~(1 << u)
                edge_count -= 1
                # Accept worsening with probability
                # Delta E = -1
                # prob = exp(-1/T) ? No, we want to maximize.
                # So worsening is bad.
                # We accept it to escape local optima.
                # If T is high, we accept more.
                # Actually, since we always accept adds, we only need to accept removes 
                # if they lead to a better state later.
                # Simple SA: accept with prob exp((E_new - E_old)/T)
                # E_new = E_old - 1. Diff = -1.
                # prob = exp(-1/T)
                if random.random() > math.exp(-1.0 / T):
                    # Reject, put it back
                    mask[u] |= (1 << v)
                    mask[v] |= (1 << u)
                    edge_count += 1
                    
        T *= cooling
        
    # Reconstruct
    edges = []
    for u in range(n):
        m = best_mask[u]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if v > u:
                edges.append([u, v])
            m ^= lsb
    return edges
