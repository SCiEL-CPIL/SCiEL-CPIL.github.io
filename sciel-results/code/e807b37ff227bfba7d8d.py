import time
import random
import itertools
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U, dtype=np.int64)
    # Compute all sums (including i=j)
    sums = np.add.outer(U_arr, U_arr)
    unique_sums = np.unique(sums)
    # Compute all differences (including i=j which gives 0)
    diffs = np.subtract.outer(U_arr, U_arr)
    unique_diffs = np.unique(diffs)
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def exhaustive_search(max_n, max_val, deadline):
    best_score = -1
    best_set = [0, 1, 3]
    
    # We include 0 in the set, so we choose n-1 elements from 1..max_val
    for n in range(3, max_n + 1):
        for comb in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                return best_score, best_set
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_score, best_set

def simulated_annealing(start_set, deadline):
    current = start_set[:]
    current_score = score_set(current)
    if current_score < 0:
        return -1, start_set
        
    best_score = current_score
    best_set = current[:]
    
    temp = 0.5
    cooling = 0.99
    min_temp = 0.01
    
    while time.time() < deadline and temp > min_temp:
        # Generate neighbor
        neighbor = current[:]
        move_type = random.random()
        
        if move_type < 0.7:
            # Shift an element
            i = random.randint(1, len(neighbor) - 1)
            delta = random.randint(-5, 5)
            if delta == 0: continue
            new_val = neighbor[i] + delta
            if new_val <= 0: continue
            neighbor[i] = new_val
            neighbor = sorted(set(neighbor))
            if len(neighbor) != len(current): continue
        elif move_type < 0.85:
            # Add an element
            if len(neighbor) < 10:
                new_val = random.randint(1, max(neighbor) + 10)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
        else:
            # Remove an element
            if len(neighbor) > 3:
                i = random.randint(1, len(neighbor) - 1)
                neighbor.pop(i)
        
        new_score = score_set(neighbor)
        
        if new_score > current_score:
            current = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
        else:
            # Accept with probability exp((new - current) / temp)
            # Since score can be negative (invalid), we handle that
            if new_score < 0:
                # Don't accept invalid states usually, but maybe allow rare jumps?
                # Better to stick to valid states for annealing
                pass
            else:
                delta_s = new_score - current_score
                if random.random() < pow(2.718, delta_s / temp):
                    current = neighbor
                    current_score = new_score
                    
        temp *= cooling
        
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin before 150s limit
    
    best_score = -1
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets
    # With numpy acceleration, we can search larger spaces
    # We go up to max_val=80 and n=7
    try:
        s1, set1 = exhaustive_search(7, 80, deadline - 20)
        if s1 > best_score:
            best_score = s1
            best_set = set1
    except Exception:
        pass
        
    # Phase 2: Simulated Annealing starting from best found
    if time.time() < deadline - 5:
        try:
            s2, set2 = simulated_annealing(best_set, deadline - 2)
            if s2 > best_score:
                best_score = s2
                best_set = set2
        except Exception:
            pass
            
    return best_set

# EVOLVE-BLOCK-END
