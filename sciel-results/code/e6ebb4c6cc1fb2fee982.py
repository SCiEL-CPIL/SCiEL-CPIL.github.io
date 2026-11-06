import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15
    all_sets = []
    for i in range(1, 17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    mask = (1 << (i-1)) | (1 << (j-1)) | (1 << (k-1)) | (1 << (l-1))
                    all_sets.append(mask)
    
    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        # Shuffle to randomize the greedy order
        random.shuffle(all_sets)
        
        current_family = []
        
        for s in all_sets:
            # Check if s can be added to current_family
            valid = True
            n = len(current_family)
            
            # Check against all pairs in current_family
            for i in range(n):
                a = current_family[i]
                # Optimization: if a & s == 0, it's unlikely to be a sunflower core unless core is empty?
                # If core is empty, a&b=0, a&s=0, b&s=0.
                # If a&s=0, then a&s == b&s implies b&s=0.
                # And a&b must be 0.
                
                for j in range(i + 1, n):
                    b = current_family[j]
                    
                    # Check sunflower condition:
                    # a & b == a & s == b & s
                    ab = a & b
                    as_ = a & s
                    bs = b & s
                    
                    if ab == as_ and as_ == bs:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_family.append(s)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
            # Optional: If we reach a very high size, we might break early?
            # Max possible is 1820. But sunflower-free is much smaller.
            # Let's keep going to try to improve.
            
    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
