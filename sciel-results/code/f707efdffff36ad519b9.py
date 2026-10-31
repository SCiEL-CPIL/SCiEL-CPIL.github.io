import time
import math

def priority(vector, n):
    # Strategy: Prioritize vectors with fewer 2s.
    # In {0,1,2}^n, a cap set is a set with no 3-term arithmetic progression.
    # An arithmetic progression x, y, z satisfies x + z = 2y (mod 3).
    # If we look at the "2" positions, they are the ones that cause "collisions" 
    # or "winding" in the mod 3 arithmetic when combined with other vectors.
    # Vectors with fewer 2s are "simpler" and tend to form larger independent sets 
    # in the Hamming cube structure over F_3.
    # 
    # Specifically, if we map 0->0, 1->1, 2->-1 in F_3, 
    # the "weight" of 2s correlates with the distance from the all-zeros vector 
    # in a way that preserves linearity better for cap set construction.
    # 
    # Let's count the number of 2s.
    # We want to MINIMIZE the number of 2s, so we assign higher priority to lower counts.
    
    count_2 = 0
    for v in vector:
        if v == 2:
            count_2 += 1
            
    # Primary score: Inverse of count_2. 
    # To make it a "priority" where higher is better, we can use a large constant minus count_2.
    # However, we need to be careful with ties.
    
    # Let's use a multi-level heuristic:
    # 1. Minimize number of 2s.
    # 2. Among those, minimize number of 1s? Or maximize number of 0s?
    #    Actually, 0s and 1s are symmetric in some sense, but 2s are the "bad" ones for 
    #    forming 0, 1, 2 progressions if we think of them as +1 steps.
    #    Let's stick to minimizing 2s first.
    
    # Tie-breaker: Use a deterministic hash to ensure consistent ordering.
    # We can use a simple polynomial hash of the vector.
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    
    # Scale the hash to a small range [0, 1) to act as a tie-breaker.
    hash_score = h / (10**9 + 7)
    
    # The main score is dominated by the count of 2s.
    # We want lower count_2 to have higher priority.
    # So priority = -count_2 * large_factor + hash_score.
    # Let's use a factor large enough to separate different counts of 2s.
    # Max count_2 is n. So factor should be > 1.
    # Let's use 1000.0 as before.
    
    return float(-count_2 * 1000.0 + hash_score)
