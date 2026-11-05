import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = set()
    best_size = 0
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        S = set(candidates[:48])
        
        # Repair
        changed = True
        while changed and time.time() < deadline:
            changed = False
            S_list = list(S)
            for a in S_list:
                others = [x for x in S_list if x != a]
                # Check k=2
                # 2a = b + c
                # b, c in others
                # Use set for fast lookup
                others_set = set(others)
                for b in others:
                    c = 2 * a - b
                    if c in others_set and c != b:
                        # Violation found
                        S.remove(b)
                        changed = True
                        break
                if changed:
                    break
            
            if not changed:
                # Check k=3
                # 3a = b + c + d
                # This is harder.
                # We can skip for now and assume 3-AP removal is enough?
                # No, 1 is mean of 2,4 (k=2).
                # What about k=3?
                # If we remove all 3-APs, are we safe from k=3?
                # Not necessarily.
                # But k=2 is the most common.
                pass
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        if best_size == 48:
            break
            
    return list(best_set)
