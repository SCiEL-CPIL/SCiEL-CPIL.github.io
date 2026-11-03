import time
import random
from math import log, log2
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s
    
    random.seed(12345)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # Strategy: Model the problem as finding a set where differences are "independent" 
    # and sums collide. This is related to Sidon sets but we want to maximize 
    # |U-U|/|U+U|. 
    
    # Key insight: If U is a geometric progression with ratio r, then:
    # |U-U| ~ n^2 (all differences distinct if r is large enough)
    # |U+U| ~ n(n+1)/2 (all sums distinct)
    # Ratio ~ 2, which is constant.
    # 
    # To get ratio > 2, we need sums to collide more than differences.
    # Consider U = {0, a, b, a+b, ...} additive subgroups or structured sets.
    
    # Let's try a "greedy difference maximizer" with sum collision awareness.
    # We want to add elements that create many new differences but few new sums.
    
    def greedy_construct(max_val, num_elems):
        """
        Greedily build a set by adding elements that maximize the ratio increment.
        """
        U = [0]
        for _ in range(num_elems - 1):
            best_candidate = None
            best_delta_score = -1
            
            # Candidate pool: sample from range
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            candidates = candidates[:200]  # Limit search space for speed
            
            for c in candidates:
                if c in U:
                    continue
                new_U = sorted(U + [c])
                s = fast_score(new_U)
                if s > 0 and s > best_delta_score:
                    best_delta_score = s
                    best_candidate = c
            
            if best_candidate is not None:
                U.append(best_candidate)
            else:
                break
        
        return U
    
    # Run greedy construction with different parameters
    for max_val in [50, 100, 200, 500]:
        for num_elems in [5, 6, 7, 8, 9, 10]:
            if time.time() > deadline - 20:
                break
            U = greedy_construct(max_val, num_elems)
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Local search / Hill climbing on the best set found
    def local_search(U, max_iter=5000):
        current = sorted(set(U))
        current_score = fast_score(current)
        if current_score < 0:
            return current, current_score
        
        best = current
        best_score_local = current_score
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            n = len(current)
            # Move: replace one element
            if n < 2:
                break
                
            i = random.randint(1, n-1)  # Don't remove 0
            old_val = current[i]
            max_val = max(current)
            
            # Propose new value
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-max(1, int(max_val*0.1)), max(1, int(max_val*0.1)))
                new_val = old_val + delta
            else:
                # Random in range
                new_val = random.randint(1, max_val * 2)
            
            if new_val <= 0 or new_val in current:
                continue
            
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != n:
                continue
                
            new_score = fast_score(new_set)
            
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best = current
            else:
                # Accept worse with small probability (simulated annealing lite)
                if random.random() < 0.1:
                    current = new_set
                    current_score = new_score
        
        return best, best_score_local
    
    # Run local search on best set
    if time.time() < deadline - 10:
        refined, refined_score = local_search(best_set, max_iter=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Try a few random restarts with greedy + local search
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        
        # Random initial set
        n = random.randint(4, 8)
        max_v = random.randint(50, 300)
        U_init = [0] + sorted(random.sample(range(1, max_v), n-1))
        
        # Local search from random start
        refined, refined_score = local_search(U_init, max_iter=5000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
