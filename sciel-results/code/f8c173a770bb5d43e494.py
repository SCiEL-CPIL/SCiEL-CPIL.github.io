"""
Strategy: Deterministic search over geometric progressions.
The score depends on the ratio |U-U|/|U+U| and the maximum element.
Geometric progressions {1, r, r^2, ..., r^{k-1}} have known sum/difference set properties.
We search for integer base r and length k that maximize the score.
"""
import random
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        
        # Compute sumset and difference set efficiently for small sets
        # For geometric sets, we can compute these directly or use sets
        sumset = set()
        diffset = set()
        
        # Optimization: If set is large, this is slow. 
        # But geometric sets with reasonable k are small enough.
        # We will limit k to keep this fast.
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        
        return exponent

    best_set = [0, 1]
    best_score = 0.0
    
    # Search over geometric progressions: 1, r, r^2, ..., r^{k-1}
    # Note: The problem requires non-negative integers containing 0.
    # A standard geometric progression starts at 1. We can add 0.
    # U = {0} union {1, r, r^2, ...}
    
    # Let's explore bases r and lengths k.
    # r=2 is binary-like. r=3 is ternary-like.
    # Larger r makes max U grow faster, which hurts the denominator log(2 max U + 1).
    # But it might increase the ratio |U-U|/|U+U|.
    
    # Let's test a range of r and k.
    # Constraint: 120s limit. score_set is O(k^2).
    # If k=20, k^2=400 ops. Very fast.
    # We can test many combinations.
    
    max_k = 30
    max_r = 50
    
    for r in range(2, max_r + 1):
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        current_power = 1
        U = [0, 1]
        
        for k in range(1, max_k + 1):
            # Add next term r^k
            if current_power > 10**18: # Prevent huge numbers
                break
                
            next_term = current_power * r
            if next_term > 10**18:
                break
                
            U.append(next_term)
            current_power = next_term
            
            # Calculate score
            s = score_set(U)
            
            if s > best_score:
                best_score = s
                best_set = list(U)
                
            # Early termination if score starts dropping significantly?
            # For geometric sets, usually there's an optimal k for a given r.
            # We can just let it run to max_k since it's fast.
            
    # Also consider arithmetic progressions?
    # U = {0, d, 2d, ..., (k-1)d}
    # This is just scaling {0, 1, ..., k-1} by d.
    # max U = (k-1)d.
    # |U+U| = 2k-1 (sums 0..2k-2).
    # |U-U| = 2k-1 (diffs -(k-1)..k-1).
    # Ratio = 1.
    # Score = 1 + log(1)/log(...) = 1.
    # This is the baseline. Geometric sets should do better if ratio > 1.
    
    # What about mixed radices or other structures?
    # For now, geometric is a strong structural candidate.
    
    # Let's also try sets of the form {0, 1, B, B^2, ...} with different B.
    # This is covered by the loop above.
    
    # Try to refine with local search on the best geometric set found?
    # Maybe not necessary if the geometric structure is optimal.
    # But let's do a quick local search on the best found set to see if we can improve.
    
    # Local search: add/remove elements near existing ones.
    # This might break the geometric structure but could find a better hybrid.
    
    current_best = list(best_set)
    current_score = best_score
    
    # Simple local search
    steps = 0
    max_steps = 5000
    
    while steps < max_steps and time.time() < deadline - 10.0:
        steps += 1
        if not current_best:
            break
            
        # Try adding a random element
        max_val = max(current_best)
        # Candidate range: up to 2*max_val
        if max_val == 0:
            cand = 1
        else:
            # Focus on gaps or extensions
            if random.random() < 0.5:
                cand = random.randint(1, max_val * 2)
            else:
                # Try extending the geometric progression further
                if len(current_best) > 1:
                    ratio_est = current_best[-1] / current_best[-2] if current_best[-2] != 0 else 2
                    cand = int(current_best[-1] * ratio_est)
                    if cand == 0:
                        cand = 1
                else:
                    cand = random.randint(1, 10)
                    
        if cand in current_best:
            continue
            
        new_set = set(current_best)
        new_set.add(cand)
        new_list = list(new_set)
        
        new_score = score_set(new_list)
        
        if new_score > current_score:
            current_score = new_score
            current_best = new_list
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_best)
        else:
            # Occasionally try removing an element (simulated annealing style)
            if random.random() < 0.1:
                removable = [x for x in current_best if x != 0]
                if removable:
                    elem = random.choice(removable)
                    new_set = set(current_best)
                    new_set.discard(elem)
                    new_list = list(new_set)
                    new_score = score_set(new_list)
                    if new_score > current_score:
                        current_score = new_score
                        current_best = new_list
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_best)

    return best_set
# EVOLVE-BLOCK-END
