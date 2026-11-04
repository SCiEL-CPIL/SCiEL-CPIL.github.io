import math
import random

def score_set(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    best_score = -1
    best_set = [0, 1, 3] # Fallback
    
    # Heuristic Greedy Construction
    # We will run multiple greedy trials with different random seeds/initializations
    
    max_elements = 30
    max_value_limit = 200
    
    for trial in range(50):
        # Start with {0}
        current_set = [0]
        
        # Randomly choose initial step or just start adding
        # To diversify, we can try adding small numbers first or random numbers
        
        # Let's try a greedy approach where we pick the best next element from a pool
        # The pool grows as we go, but we limit search space to keep it fast
        
        for step in range(max_elements - 1):
            best_candidate = None
            best_candidate_score = score_set(current_set)
            
            # Candidate pool: 
            # 1. Small integers (1 to 10)
            # 2. Integers near the current max (max_u +/- 10)
            # 3. Random integers up to max_value_limit
            
            current_max = max(current_set)
            candidates = set()
            
            # Small numbers
            for i in range(1, 15):
                candidates.add(i)
                
            # Near current max
            for i in range(max(1, current_max - 5), current_max + 15):
                candidates.add(i)
                
            # Randoms
            for _ in range(20):
                candidates.add(random.randint(1, max_value_limit))
                
            # Ensure 0 is not in candidates (already in set)
            candidates.discard(0)
            
            # Evaluate candidates
            for c in candidates:
                if c in current_set:
                    continue
                    
                new_set = current_set + [c]
                # Quick check: if max increases too much, score might drop, but let's check
                s = score_set(new_set)
                
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = c
                    
            if best_candidate is not None:
                current_set.append(best_candidate)
                if best_candidate_score > best_score:
                    best_score = best_candidate_score
                    best_set = current_set[:]
            else:
                # No improvement found, stop this trial
                break
                
    # Final check
    final_score = score_set(best_set)
    if final_score > best_score:
        best_score = final_score
        
    return best_set
