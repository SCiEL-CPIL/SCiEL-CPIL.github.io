import random
import time
from bisect import bisect_left

# EVOLVE-BLOCK-START
def construct():
    best_seq = []
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # To ensure we get a good baseline, run the original greedy first
    # but the randomized one should be better.
    
    while time.time() < deadline:
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        seq = []
        prods = set()
        
        for x in candidates:
            # Pruning: if we can't beat best, stop this trial
            if len(seq) + len(candidates) <= len(best_seq):
                break
            
            if can_add(seq, prods, x):
                add_to_seq(seq, prods, x)
        
        if len(seq) > len(best_seq):
            best_seq = seq
    
    return best_seq

def can_add(seq, prods, x):
    if not seq:
        return True
    
    # Find insertion point
    j = bisect_left(seq, x)
    
    # Check if x is already in seq (should not happen as we pick from candidates)
    # But just in case
    if j < len(seq) and seq[j] == x:
        return False
        
    m = len(seq)
    
    # Calculate L and R arrays
    # L[d] = product of seq[j-d-1...j-1] * x ? No.
    # Let's define:
    # L[d] for d in 0..j: product of elements from index j-d to j in new_seq.
    # new_seq[j] is x.
    # L[0] = x
    # L[1] = seq[j-1] * x
    # ...
    # L[d] = (product of seq[j-d..j-1]) * x
    
    # R[e] for e in 0..m-1-j: product of elements from index j to j+e in new_seq.
    # R[0] = x
    # R[1] = x * seq[j]
    # ...
    # R[e] = x * (product of seq[j..j+e-1])
    
    L = [0] * (j + 1)
    L[0] = x
    for d in range(1, j + 1):
        L[d] = L[d-1] * seq[j-d]
        
    R = [0] * (m - j + 1)
    R[0] = x
    for e in range(1, m - j + 1):
        R[e] = R[e-1] * seq[j+e-1]
        
    new_prods = set()
    
    for d in range(j + 1):
        for e in range(m - j + 1):
            # Product is L[d] * R[e] // x
            p = (L[d] * R[e]) // x
            
            if p in new_prods:
                return False
            if p in prods:
                return False
            
            new_prods.add(p)
            
    return True

def add_to_seq(seq, prods, x):
    j = bisect_left(seq, x)
    
    m = len(seq)
    
    # Recalculate L and R to update prods
    # We can just reuse the logic or recompute.
    # Recomputing is safer and fast enough.
    
    L = [0] * (j + 1)
    L[0] = x
    for d in range(1, j + 1):
        L[d] = L[d-1] * seq[j-d]
        
    R = [0] * (m - j + 1)
    R[0] = x
    for e in range(1, m - j + 1):
        R[e] = R[e-1] * seq[j+e-1]
        
    for d in range(j + 1):
        for e in range(m - j + 1):
            p = (L[d] * R[e]) // x
            prods.add(p)
            
    seq.insert(j, x)
# EVOLVE-BLOCK-END
