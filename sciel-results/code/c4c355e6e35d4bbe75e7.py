import time
import sys
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    N = 17
    MIN_D = 6
    
    # Precompute weights for all 2^N words
    max_val = 1 << N
    weights = [0] * max_val
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # Phase 1: Construct a Linear Code
    basis = []
    current_code = [0]
    
    for g in range(1, max_val):
        if time.time() > deadline - 40:
            break
            
        valid = True
        for c in current_code:
            if weights[g ^ c] < MIN_D:
                valid = False
                break
                
        if valid:
            basis.append(g)
            old_code = current_code
            current_code = []
            for c in old_code:
                current_code.append(c)
                current_code.append(c ^ g)
                
            if len(basis) >= 8:
                break
                
    # Phase 2: Non-linear Expansion via Multi-Restart Greedy
    base_code = current_code
    base_set = set(base_code)
    
    # Pre-compute scores for all candidates against the base code
    candidates = []
    
    for x in range(max_val):
        if x in base_set:
            continue
            
        min_d = max_val
        valid_candidate = True
        for c in base_code:
            d = weights[x ^ c]
            if d < MIN_D:
                valid_candidate = False
                break
            if d < min_d:
                min_d = d
                
        if valid_candidate:
            candidates.append((min_d, x))
            
    # Group candidates by score to allow shuffling within ties
    score_groups = defaultdict(list)
    for score, x in candidates:
        score_groups[score].append(x)
        
    sorted_scores = sorted(score_groups.keys(), reverse=True)
    
    best_result = base_code[:]
    best_size = len(base_code)
    
    # Multi-Restart Greedy
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        added_in_phase2 = []
        existing_set = set(base_code)
        
        for s in sorted_scores:
            group = score_groups[s][:]
            random.shuffle(group)
            
            for x in group:
                # Check against added_in_phase2
                valid = True
                for c in added_in_phase2:
                    if weights[x ^ c] < MIN_D:
                        valid = False
                        break
                        
                if valid:
                    added_in_phase2.append(x)
                    existing_set.add(x)
                    
        current_result = base_code + added_in_phase2
        if len(current_result) > best_size:
            best_size = len(current_result)
            best_result = current_result[:]
            
    return best_result
