"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
_LCM_ALL = 30030

def _backtrack(choices, assignment, used, idx):
    if idx == N:
        return True
    for val in choices[idx]:
        if val not in used:
            used.add(val)
            assignment[idx] = val
            if _backtrack(choices, assignment, used, idx + 1):
                return True
            used.remove(val)
            assignment[idx] = None
    return False

def _find_best():
    best_span = float('inf')
    best_assignment = None
    
    deadline = time.time() + 80.0
    
    for W in range(1, 50):
        if time.time() > deadline:
            break
        
        for L in range(_LCM_ALL):
            if time.time() > deadline:
                return best_span, best_assignment
            
            choices = []
            feasible = True
            for i in range(N):
                d = DIVISORS[i]
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    feasible = False
                    break
                choices.append(list(range(k_min * d, k_max * d + 1, d)))
            
            if not feasible:
                continue
            
            assignment = [None] * N
            used = set()
            if _backtrack(choices, assignment, used, 0):
                span = max(assignment) - min(assignment)
                if span < best_span:
                    best_span = span
                    best_assignment = assignment[:]
                    if span == 1:  # Can't do better than 1 for distinct integers
                        return best_span, best_assignment
    
    return best_span, best_assignment

def construct():
    best_span, best_assignment = _find_best()
    
    if best_assignment is None:
        return DIVISORS[:]
    
    return [int(x) for x in best_assignment]

# EVOLVE-BLOCK-END
