import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # 10s margin
    
    def is_valid(residue, modulus, current_classes):
        for r_i, m_i in current_classes:
            g = math.gcd(modulus, m_i)
            if (residue - r_i) % g == 0:
                return False
        return True

    def greedy_restart(order):
        classes = []
        for m in order:
            # Try residues
            # Heuristic: try residues that are "far" from existing ones?
            # Just try all
            for r in range(m):
                if is_valid(r, m, classes):
                    classes.append([r, m])
                    break
        return classes

    best_sol = []
    best_len = 0
    
    # Try sorted
    sol = greedy_restart(list(range(2, 129)))
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol
        
    # Try reversed
    sol = greedy_restart(list(range(128, 1, -1)))
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol
        
    # Random restarts
    for _ in range(1000):
        if time.time() > deadline:
            break
        order = list(range(2, 129))
        random.shuffle(order)
        sol = greedy_restart(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol
            
    # If best_sol is empty (should not happen), return a known valid one
    if not best_sol:
        return [[0, 2], [1, 4], [3, 8], [7, 16], [15, 32], [31, 64], [63, 128]]
        
    return best_sol
