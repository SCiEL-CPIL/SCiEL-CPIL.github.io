import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 115  # 5s safety margin
    
    n, m, p = 2, 2, 12
    
    # First, build the known rank-42 solution
    best_rank = 42
    best_solution = _build_rank42(n, m, p)
    
    # Try to find lower rank
    for target_rank in range(36, 42):
        if time.time() > deadline - 10:
            break
        sol = _search_rank(n, m, p, target_rank, time_limit=8)
        if sol is not None:
            best_rank = target_rank
            best_solution = sol
            break
    
    return best_solution

def _build_rank42(n, m, p):
    # ... same as current ...
    
def _search_rank(n, m, p, target_rank, time_limit):
    # Try random initializations and solve
    ...
