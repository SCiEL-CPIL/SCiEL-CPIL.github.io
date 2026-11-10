# EVOLVE-BLOCK-START
import random
import time

def construct():
    # Use a deterministic greedy to start
    def greedy_smallest():
        chosen = []
        counts = {}
        for x in range(1, 1001):
            new_sums = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            ok = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                counts = trial
        return chosen, counts

    # Local search to improve
    def local_search(initial_set, initial_counts, max_time=110):
        S = set(initial_set)
        counts = dict(initial_counts)
        start_time = time.time()
        no_improvement_count = 0
        
        while time.time() - start_time < max_time:
            # Try to add a random element
            candidates = [x for x in range(1, 1001) if x not in S]
            if not candidates:
                break
            
            x = random.choice(candidates)
            
            # Check if x can be added
            conflict_y = None
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    conflict_y = y
                    break
            if 2 * x in counts and counts[2 * x] >= 2:
                # Self-sum conflict, but we don't have a specific y to remove easily 
                # unless we remove x? No, x is not in S.
                # If 2x is already at 2, we can't add x.
                # We need to find a y such that removing y helps.
                # But 2x doesn't involve y from S.
                # So if 2x is full, we must remove an element that contributes to 2x?
                # Only x itself contributes to 2x. But x is not in S.
                # So if 2x is full, we cannot add x unless we remove something that *is* x? No.
                # Wait, if 2x is full, it means there are already two pairs (a,b) in S such that a+b=2x.
                # Adding x creates a third pair (x,x).
                # To add x, we must break one of the existing pairs.
                # So we need to remove an element involved in those pairs.
                
                # For simplicity, if direct add fails, we look for a swap.
                pass
            
            if conflict_y is None and (2 * x not in counts or counts[2 * x] < 2):
                # Can add x
                # Update counts
                for y in S:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                S.add(x)
                no_improvement_count = 0
            else:
                # Try to swap: remove a random y from S, then add x
                # We need to pick a y that resolves the conflict.
                # If conflict was x+y >= 2, removing y helps.
                # If conflict was 2x >= 2, removing any y involved in 2x sums helps.
                
                # Let's just try a random y from S
                if S:
                    y_remove = random.choice(list(S))
                    
                    # Check if removing y_remove and adding x is valid
                    # 1. Remove y_remove from counts
                    trial_counts = dict(counts)
                    for z in S:
                        if z == y_remove: continue
                        s = y_remove + z
                        trial_counts[s] -= 1
                        if trial_counts[s] == 0: del trial_counts[s]
                    s_self = 2 * y_remove
                    trial_counts[s_self] -= 1
                    if trial_counts[s_self] == 0: del trial_counts[s_self]
                    
                    # 2. Add x to trial_counts
                    ok = True
                    for z in S:
                        if z == y_remove: continue
                        s = x + z
                        if trial_counts.get(s, 0) >= 2:
                            ok = False
                            break
                    if ok:
                        s_self_x = 2 * x
                        if trial_counts.get(s_self_x, 0) >= 2:
                            ok = False
                    if ok:
                        # Apply swap
                        # Remove y_remove from actual counts
                        for z in S:
                            if z == y_remove: continue
                            s = y_remove + z
                            counts[s] -= 1
                            if counts[s] == 0: del counts[s]
                        s_self = 2 * y_remove
                        counts[s_self] -= 1
                        if counts[s_self] == 0: del counts[s_self]
                        S.remove(y_remove)
                        
                        # Add x
                        for z in S:
                            s = x + z
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x] = counts.get(2 * x, 0) + 1
                        S.add(x)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1

            if no_improvement_count > 1000:
                break

        return list(S)

    # Initialize
    init_set, init_counts = greedy_smallest()
    result = local_search(init_set, init_counts)
    
    return result
# EVOLVE-BLOCK-END
