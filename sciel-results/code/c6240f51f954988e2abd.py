import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85
    
    # 1. Empty set
    family = [0]
    
    # 2. All singletons (size 1)
    for i in range(n):
        family.append(1 << i)
        
    # 3. Full set (size 12)
    full_mask = (1 << n) - 1
    family.append(full_mask)
    
    # 4. All sets of size 11
    for i in range(n):
        # Remove bit i from full set
        family.append(full_mask ^ (1 << i))
        
    # 5. 27 sets of size 5
    # We just need any 27 distinct sets of size 5
    # Generate all size 5 sets and take the first 27
    count = 0
    target = 27
    # Iterate through all masks and pick those with popcount 5
    # Or generate combinations
    import itertools
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 5):
        if count >= target:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        family.append(mask)
        count += 1
        
    # Convert to list of lists
    result = []
    for mask in family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
