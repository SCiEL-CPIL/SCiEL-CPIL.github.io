"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    deadline = time.time() + 55  # 5s safety margin
    
    # Fast cost function: compute max absolute sum over all arithmetic progressions
    def get_cost(seq):
        max_abs = 0
        s = seq
        nn = n
        for step in range(1, nn):
            for start in range(nn):
                curr = 0
                idx = start
                while idx < nn:
                    curr += s[idx]
                    if curr > max_abs:
                        max_abs = curr
                    elif curr < -max_abs:
                        max_abs = -curr
                    idx += step
        return max_abs
    
    # Initialize with random signs
    rng = random.Random(42)
    seq = [rng.choice((-1, 1)) for _ in range(n)]
    best_cost = get_cost(seq)
    best_seq = list(seq)
    
    # Local search: flip one position at a time, accept if it reduces max cost
    while time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(n):
            # Quick check: only flip if it might help
            seq[i] = -seq[i]
            cost = get_cost(seq)
            if cost < best_cost:
                best_cost = cost
                best_seq = list(seq)
                improved = True
            else:
                seq[i] = -seq[i]  # revert
        
        if not improved:
            # Stuck in local minimum, do a random restart with perturbation
            # Perturb best_seq by flipping a few random positions
            seq = list(best_seq)
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                pos = rng.randint(0, n - 1)
                seq[pos] = -seq[pos]
            best_cost = get_cost(seq)
    
    return best_seq
# EVOLVE-BLOCK-END
