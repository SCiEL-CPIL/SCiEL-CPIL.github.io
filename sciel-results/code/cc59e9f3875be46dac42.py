import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Known valid set of size 9
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def backtrack(current_set, S, sum_card, used, candidates, target_size):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        if len(current_set) == target_size:
            return
            
        # Try candidates
        for i in range(len(candidates)):
            x = candidates[i]
            if x in used:
                continue
            
            # Check validity
            k = len(current_set)
            valid = True
            added_keys = []
            for s in S[k]:
                v = s + x
                if v in sum_card:
                    if sum_card[v] != k + 1:
                        valid = False
                        break
                else:
                    added_keys.append(v)
            
            if not valid:
                continue
                
            # Commit
            used.add(x)
            current_set.append(x)
            
            new_sum_set = set()
            for s in S[k]:
                v = s + x
                new_sum_set.add(v)
                sum_card[v] = k + 1
                
            S.append(new_sum_set)
            
            # Recurse with remaining candidates
            # To avoid copying candidates list, we can pass indices or a new list.
            # Copying is O(N). N=256. Depth 10. 2560 ops per path.
            # Acceptable.
            remaining = candidates[:i] + candidates[i+1:]
            
            backtrack(current_set, S, sum_card, used, remaining, target_size)
            
            # Undo
            S.pop()
            for v in added_keys:
                del sum_card[v]
            current_set.pop()
            used.remove(x)
            
            if time.time() > deadline:
                return

    # Try to find larger sets
    for target_size in [12, 11, 10]:
        found_better = False
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Reset state
            S = [{0}]
            sum_card = {0: 0}
            used = set()
            current_set = []
            
            # Randomize search order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            backtrack(current_set, S, sum_card, used, candidates, target_size)
            
            if len(best_set) >= target_size:
                found_better = True
                break
                
        if found_better:
            break
            
    return best_set
