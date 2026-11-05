import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    mandatory = [1, 2, 5, 11]
    mandatory_set = set(mandatory)
    
    best_set = mandatory.copy()
    best_size = len(best_set)
    
    # Bitset for sums. Max sum 284+284=568.
    # We need bits 0..568.
    
    def get_initial_mask():
        mask = 0
        for i in range(len(mandatory)):
            for j in range(i, len(mandatory)):
                s = mandatory[i] + mandatory[j]
                mask |= (1 << s)
        return mask

    init_mask = get_initial_mask()
    
    def search_size(target_size, max_restarts=1000):
        nonlocal best_set, best_size
        
        candidates = [x for x in range(1, 285) if x not in mandatory_set]
        
        for r in range(max_restarts):
            if time.time() > deadline:
                return False
            
            random.shuffle(candidates)
            
            def bt(curr_S, curr_mask):
                if len(curr_S) == target_size:
                    return True
                
                # Find valid next
                valid_next = []
                curr_S_set = set(curr_S)
                
                # Optimization: 
                # We can check candidates in a specific order.
                # To pack tightly, smaller numbers are often better?
                # Let's just iterate all and collect valid ones, then pick.
                
                for x in candidates:
                    if x in curr_S_set:
                        continue
                    
                    # Check sums
                    # x + y for y in curr_S
                    # 2*x
                    ok = True
                    # Check 2*x first, it's cheap
                    if curr_mask & (1 << (2 * x)):
                        ok = False
                    else:
                        for y in curr_S:
                            s = x + y
                            if curr_mask & (1 << s):
                                ok = False
                                break
                    if ok:
                        valid_next.append(x)
                        
                if not valid_next:
                    return False
                
                # Heuristic: Pick from the smallest few (tight packing)
                valid_next.sort()
                # Pick 1 to 5 smallest randomly
                k = min(5, len(valid_next))
                choice = random.choice(valid_next[:k])
                
                new_S = curr_S + [choice]
                new_S.sort()
                
                new_mask = curr_mask
                for y in new_S:
                    s = choice + y
                    new_mask |= (1 << s)
                    
                if bt(new_S, new_mask):
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = new_S[:]
                    return True
                    
                return False

            if bt(mandatory[:], init_mask):
                return True
                
        return False

    for target in range(20, 15, -1):
        if search_size(target, max_restarts=1000):
            break
            
    return best_set
