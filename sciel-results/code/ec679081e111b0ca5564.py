import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    selected = []
    
    # Try adding numbers in descending order (large numbers are safer)
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if we can add x to selected
        can_add = True
        
        # Check 1: x should not divide sum of any 3 distinct elements in selected
        if len(selected) >= 3:
            for tri in combinations(selected, 3):
                if (tri[0] + tri[1] + tri[2]) % x == 0:
                    can_add = False
                    break
        
        if not can_add:
            continue
        
        # Check 2: For each a in selected, a should not divide (x + s1 + s2)
        # where s1, s2 are distinct elements from selected excluding a
        if len(selected) >= 2:
            for i, a in enumerate(selected):
                # Get all elements in selected except a
                others = [v for j, v in enumerate(selected) if j != i]
                # Check all pairs from others
                for s1_idx in range(len(others)):
                    s1 = others[s1_idx]
                    for s2_idx in range(s1_idx + 1, len(others)):
                        s2 = others[s2_idx]
                        if (x + s1 + s2) % a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                if not can_add:
                    break
        
        if can_add:
            selected.append(x)
    
    return selected
