import random
import time

def construct():
    n = 256
    
    # Helper to count 3-APs efficiently for a set
    def count_3aps(S):
        count = 0
        S_list = sorted(S)
        # For each pair (a, c), check if (a+c)/2 is in S
        # This is O(|S|^2). |S| ~ 100-200. 200^2 = 40000. Fast enough.
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                c = S_list[j]
                s = a + c
                if s % 2 == 0:
                    mid = s // 2
                    if mid in S:
                        count += 1
        return count

    def has_4ap(S, x, exclude=None):
        # Check if x in S creates a 4-AP with other elements in S
        # If exclude is provided, ignore 4-APs that contain elements in exclude? No, just check if adding x creates a NEW 4-AP.
        # We check if there exists a, b, c in S\{x} such that {x, a, b, c} is a 4-AP.
        # Or simply, check all 4-APs involving x.
        for d in range(1, n+1):
            # x is first
            if x+d in S and x+2*d in S and x+3*d in S:
                return True
            # x is second
            if x-d >= 1 and x-d in S and x+d in S and x+2*d in S:
                return True
            # x is third
            if x-2*d >= 1 and x-2*d in S and x-d in S and x+d in S:
                return True
            # x is fourth
            if x-3*d >= 1 and x-3*d in S and x-2*d in S and x-d in S:
                return True
        return False

    def is_valid(S):
        # Check if S has any 4-AP
        # Optimization: Only check 4-APs where the max element is in S?
        # Just iterate all possible 4-APs in [1, n]
        for a in range(1, n+1):
            for d in range(1, (n-a)//3 + 1):
                b = a+d
                c = a+2*d
                e = a+3*d
                if a in S and b in S and c in S and e in S:
                    return False
        return True

    # Phase 1: Generate a valid initial set
    # Use greedy removal to get a valid set quickly
    S = set(range(1, n+1))
    changed = True
    while changed:
        changed = False
        found = None
        for a in range(1, n+1):
            for d in range(1, (n-a)//3 + 1):
                b = a+d
                c = a+2*d
                e = a+3*d
                if a in S and b in S and c in S and e in S:
                    found = (a, b, c, e)
                    break
            if found:
                break
        if found:
            a, b, c, e = found
            # Remove a random element from the 4-AP to break it
            # Or remove the one with lowest 3-AP degree? 
            # For initialization, random is fine to get diverse starts
            victim = random.choice([a, b, c, e])
            S.remove(victim)
            changed = True
    
    # Phase 2: Local Search (Hill Climbing)
    # Try to improve 3-AP count by swapping elements
    best_score = count_3aps(S)
    current_score = best_score
    no_improve_count = 0
    time_limit = time.time() + 50  # 50 seconds limit
    
    while time.time() < time_limit:
        if no_improve_count > 100:
            # Perturb
            # Remove a random element from S
            if S:
                x = random.choice(list(S))
                S.remove(x)
            # Add a random element not in S
            candidates = [i for i in range(1, n+1) if i not in S]
            if candidates:
                y = random.choice(candidates)
                if not has_4ap(S, y):
                    S.add(y)
            no_improve_count = 0
            continue
        
        # Pick a random element to remove
        x = random.choice(list(S))
        # Pick a random element to add
        candidates = [i for i in range(1, n+1) if i not in S]
        if not candidates:
            break
        y = random.choice(candidates)
        
        # Try swap x -> y
        # Check if removing x and adding y maintains validity
        # 1. Remove x
        S.remove(x)
        # 2. Check if y can be added
        valid = not has_4ap(S, y)
        
        if valid:
            S.add(y)
            new_score = count_3aps(S)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                no_improve_count = 0
            else:
                # Revert
                S.remove(y)
                S.add(x)
                no_improve_count += 1
        else:
            # Revert
            S.add(x)
            no_improve_count += 1
            
    # Phase 3: Final polish - try adding any element that doesn't break validity
    for y in range(1, n+1):
        if y not in S and not has_4ap(S, y):
            S.add(y)
            # Check if it actually increases score? If it doesn't, maybe remove?
            # But adding an element always adds 3-APs (or keeps same if it doesn't form any).
            # It can only increase or keep the count. It never decreases 3-APs.
            # Wait, adding an element x to S:
            # New 3-APs are those involving x.
            # Old 3-APs remain.
            # So count_3aps(S U {x}) >= count_3aps(S).
            # So we should always add if valid.
            
    return sorted(S)
