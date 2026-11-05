"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    P = 101
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]
    
    # For each residue, track the minimum-size subset achieving it
    best_size = [11] * P  # initialize with impossible size
    best_subset = [None] * P
    
    # Iterate all non-empty subsets of {1..10} via bitmask
    for mask in range(1, 1 << 10):
        s = 0
        subset = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                s += inv[x]
                subset.append(x)
        r = s % P
        sz = len(subset)
        if sz < best_size[r]:
            best_size[r] = sz
            best_subset[r] = subset
    
    # Verify all residues are covered
    # (should always be true given 1023 subsets for 101 residues)
    for r in range(P):
        if best_subset[r] is None:
            # Fallback: use any subset (should not happen)
            for mask in range(1, 1 << 10):
                s = 0
                subset = []
                for x in range(1, 11):
                    if mask & (1 << (x - 1)):
                        s += inv[x]
                        subset.append(x)
                if s % P == r:
                    best_subset[r] = subset
                    best_size[r] = len(subset)
                    break
    
    # Check time
    if time.time() > deadline:
        pass  # Should not happen; this computation is O(1023 * 10)
    
    return [best_subset[r] for r in range(P)]
# EVOLVE-BLOCK-END
