import time
import math
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # 1. Greedy Construction
    # Candidates are 1..511. We want max 1/(512-a), so iterate a from 511 down to 1.
    chosen = []
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    # Helper to calculate score
    def score_set(s):
        return sum(1.0 / (512 - a) for a in s)
    
    current_score = score_set(chosen)
    
    # 2. Local Search: 1-for-1 Swaps
    # Try to improve by removing one element and adding another.
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Sort current set to maintain some order, though not strictly necessary
        # We iterate over each element in the current set to see if removing it allows a better addition
        for i in range(len(chosen)):
            if (time.time() - start_time) > time_limit:
                break
                
            a_remove = chosen[i]
            # Temporarily remove a_remove
            remaining = chosen[:i] + chosen[i+1:]
            
            # Find the best candidate b not in chosen that is coprime to all in remaining
            best_b = -1
            best_weight = 0.0
            
            # Iterate candidates from 511 down to 1
            # Optimization: Only check b that are NOT in chosen
            # We can precompute a set of chosen for O(1) lookup
            chosen_set = set(chosen)
            
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                
                # Check if b is coprime to all in remaining
                if all(math.gcd(b, r) == 1 for r in remaining):
                    w = 1.0 / (512 - b)
                    if w > best_weight:
                        best_weight = w
                        best_b = b
                    # Since we iterate b in decreasing order (increasing weight), 
                    # the first valid b found is the best one.
                    # Wait, is 1/(512-b) monotonic with b? Yes.
                    # So the first valid b (largest b) has the highest weight.
                    break
            
            # If we found a b, check if swapping improves score
            if best_b != -1:
                # New score = current_score - weight(a_remove) + weight(best_b)
                w_remove = 1.0 / (512 - a_remove)
                if best_weight > w_remove:
                    # Perform swap
                    chosen[i] = best_b
                    current_score += (best_weight - w_remove)
                    improved = True
                    # After a swap, the set has changed. 
                    # It's safer to restart the outer loop or continue carefully.
                    # Restarting is simpler and less error-prone for correctness.
                    # However, restarting might be slow if many swaps happen.
                    # Given the small size, let's just break and restart the while loop.
                    break
        
    return chosen
