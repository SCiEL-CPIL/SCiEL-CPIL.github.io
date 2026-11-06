import time
import numpy as np

def construct():
    deadline = time.time() + 85.0 # Safety margin
    
    selected = []
    existing_sums = set()
    
    # We iterate x from 1 to 4096
    for x in range(1, 4097):
        if time.time() > deadline:
            break
            
        if not selected:
            selected.append(x)
            existing_sums.add(3 * x)
            continue
            
        # Convert selected to numpy array for vectorized ops
        sel = np.array(selected, dtype=np.int32)
        n = len(sel)
        
        # Generate new sums
        # 1. 3x
        # 2. 2x + sel
        # 3. x + 2*sel
        # 4. x + sel[i] + sel[j] for i <= j
        
        # Pre-allocate or use concatenate
        # Size: 1 + n + n + n(n+1)/2
        num_pairs = n * (n + 1) // 2
        total_size = 1 + 2 * n + num_pairs
        
        new_sums = np.empty(total_size, dtype=np.int32)
        
        idx = 0
        new_sums[idx] = 3 * x
        idx += 1
        
        new_sums[idx:idx+n] = 2 * x + sel
        idx += n
        
        new_sums[idx:idx+n] = x + 2 * sel
        idx += n
        
        # Upper triangle indices
        ii, jj = np.triu_indices(n)
        new_sums[idx:] = x + sel[ii] + sel[jj]
        
        # Check internal uniqueness
        # If there are duplicates in new_sums, the triple sums are not unique among themselves
        # Note: This check is necessary because different triples involving x could sum to the same value
        # e.g., 2x + y1 == x + 2y2 + ... wait, no, the types are distinct by construction?
        # No, 2x+y is one type, x+2y is another.
        # Can 2x+y == x+2y? => x == y. If y is in selected, and x is candidate, x != y (since x is new and larger? No, x is just the next candidate in range).
        # Actually, x is distinct from all y in selected.
        # Can 2x+y1 == 2x+y2? => y1==y2. Distinct y's.
        # Can x+2y1 == x+2y2? => y1==y2.
        # Can 2x+y == x+y+z? => x == z. If z is in selected, x is not.
        # So, are there *any* internal collisions?
        # The problem statement says "Every unordered triple... must have a different sum".
        # This implies the mapping (a,b,c) -> a+b+c is injective for all a<=b<=c in the final set.
        # When we add x, we are adding triples (x,x,x), (x,x,y), (x,y,y), (x,y,z).
        # We must ensure that no two of *these* new triples have the same sum.
        # AND that none of these new sums collide with *old* sums.
        
        # Is it possible for two new triples to have the same sum?
        # (x,x,x) sum 3x.
        # (x,x,y) sum 2x+y.
        # (x,y,y) sum x+2y.
        # (x,y,z) sum x+y+z.
        
        # Collision 1: 3x == 2x+y => x == y. Impossible since x is new.
        # Collision 2: 3x == x+2y => 2x == 2y => x == y. Impossible.
        # Collision 3: 3x == x+y+z => 2x == y+z. Possible!
        # Collision 4: 2x+y1 == 2x+y2 => y1==y2.
        # Collision 5: 2x+y1 == x+2y2 => x + y1 == 2y2. Possible!
        # Collision 6: 2x+y1 == x+y2+z2 => x + y1 == y2 + z2. Possible!
        # Collision 7: x+2y1 == x+2y2 => y1==y2.
        # Collision 8: x+2y1 == x+y2+z2 => 2y1 == y2+z2. Possible!
        # Collision 9: x+y1+z1 == x+y2+z2 => y1+z1 == y2+z2. Possible!
        
        # So yes, we MUST check for internal duplicates in new_sums.
        # np.unique is efficient.
        
        unique_new = np.unique(new_sums)
        if len(unique_new) != total_size:
            continue
            
        # Check against existing sums
        # Convert unique_new to set for intersection
        # This conversion is O(total_size).
        new_sums_set = set(unique_new.tolist())
        
        if existing_sums & new_sums_set:
            continue
            
        # Valid
        selected.append(x)
        existing_sums |= new_sums_set
        
    return selected
