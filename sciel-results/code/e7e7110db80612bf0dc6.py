"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Search for minimum span W
    # We assume optimal W is small.
    # We assume optimal L is small (due to shift invariance of the problem structure for small W).
    
    max_W = 500
    max_L = 2000
    
    for W in range(1, max_W + 1):
        if time.time() > deadline:
            break
            
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
                
            # Check if window [L, L+W] has at least one multiple for each divisor
            candidates = []
            feasible = True
            for d in DIVISORS:
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                cands = [k * d for k in range(start_k, end_k + 1)]
                if not cands:
                    feasible = False
                    break
                candidates.append(cands)
            
            if not feasible:
                continue
                
            # Sort candidates by length to improve backtracking pruning
            order = sorted(range(N), key=lambda i: len(candidates[i]))
            
            # Backtracking
            assigned = [None] * N
            used = set()
            
            def backtrack(idx):
                if idx == N:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assigned[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assigned[i] = None
                return False
            
            if backtrack(0):
                result = [int(assigned[i]) for i in range(N)]
                # Safety check
                if len(set(result)) == N:
                    for i, d in enumerate(DIVISORS):
                        if result[i] % d != 0:
                            return DIVISORS[:]
                    return result
    
    # Fallback
    return DIVISORS[:]
# EVOLVE-BLOCK-END
