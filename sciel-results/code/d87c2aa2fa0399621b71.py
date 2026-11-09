import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 50.0
    
    def count_pairs(family):
        # Use NumPy for speed
        fam_np = np.array(family, dtype=np.int64)
        A = fam_np[:, np.newaxis]
        B = fam_np[np.newaxis, :]
        inter = A & B
        S = (inter == A) # A subset B
        iu, ju = np.triu_indices(m, k=1)
        return int(np.sum(S[iu, ju]))

    def random_family():
        # Random subsets of {0..11}
        return [random.randint(0, (1<<n)-1) for _ in range(m)]

    def structured_family():
        # Pick a random base set of size 6
        # Generate all subsets
        # Pick 40 distinct ones
        base_elements = random.sample(range(n), 6)
        base_mask = 0
        for e in base_elements:
            base_mask |= (1 << e)
        
        # All subsets of base_mask
        # Iterate through 0..63, map to base_mask
        subsets = []
        for i in range(64):
            mask = 0
            for b in range(6):
                if (i >> b) & 1:
                    mask |= (1 << base_elements[b])
            subsets.append(mask)
        
        # Shuffle and take 40
        random.shuffle(subsets)
        return subsets[:m]

    best_family = None
    best_score = -1

    # Run several restarts
    while time.time() < deadline:
        # Alternate between random and structured starts
        if random.random() < 0.5:
            family = structured_family()
        else:
            family = random_family()
        
        # Ensure distinctness (structured is always distinct, random might not be)
        if len(set(family)) < m:
             # Fix duplicates
             seen = set()
             new_family = []
             for f in family:
                 if f not in seen:
                     new_family.append(f)
                     seen.add(f)
             while len(new_family) < m:
                 new_f = random.randint(0, (1<<n)-1)
                 if new_f not in seen:
                     new_family.append(new_f)
                     seen.add(new_f)
             family = new_family
             
        current_score = count_pairs(family)
        
        # Local Search
        while time.time() < deadline:
            improved = False
            # Try a few random moves
            for _ in range(100):
                if time.time() >= deadline:
                    break
                    
                i = random.randint(0, m-1)
                old_mask = family[i]
                
                # Generate candidate
                # 1. Flip 1 bit
                # 2. Flip 2 bits
                # 3. Random
                r = random.random()
                if r < 0.6:
                    # Flip 1 bit
                    bit = random.randint(0, n-1)
                    new_mask = old_mask ^ (1 << bit)
                elif r < 0.9:
                    # Flip 2 bits
                    b1, b2 = random.sample(range(n), 2)
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    # Random
                    new_mask = random.randint(0, (1<<n)-1)
                
                if new_mask == old_mask:
                    continue
                if new_mask in family:
                    # Check if it's the same index
                    # We need to ensure new_mask is not in family[j] for j != i
                    # Simple check:
                    if family.count(new_mask) > 1:
                        continue
                    # If family.count(new_mask) == 1, it must be family[i] which is old_mask
                    # But we already checked new_mask != old_mask
                    # So if count is 1, it's a duplicate of another? No.
                    # If new_mask is in family, and new_mask != old_mask, then it's a duplicate.
                    # So:
                    if new_mask in family:
                        continue

                family[i] = new_mask
                new_score = count_pairs(family)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    # Accept and continue loop (don't break inner loop to try more)
                else:
                    family[i] = old_mask
            
            if not improved:
                # Stuck. Perturb.
                # Replace 2 random sets with random masks
                i1, i2 = random.sample(range(m), 2)
                family[i1] = random.randint(0, (1<<n)-1)
                family[i2] = random.randint(0, (1<<n)-1)
                # Ensure distinctness after perturbation
                seen = set()
                new_family = []
                for f in family:
                    if f not in seen:
                        new_family.append(f)
                        seen.add(f)
                while len(new_family) < m:
                    new_f = random.randint(0, (1<<n)-1)
                    if new_f not in seen:
                        new_family.append(new_f)
                        seen.add(new_f)
                family = new_family
                current_score = count_pairs(family)
                
                # If score dropped significantly, break to next restart?
                # Or just continue.
                # Let's break to restart if we've been stuck for a while?
                # For simplicity, just continue.
        
        if current_score > best_score:
            best_score = current_score
            best_family = family[:]

    # Convert to list of lists of ints
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
