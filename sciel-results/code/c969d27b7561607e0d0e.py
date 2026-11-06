"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    
    def make_maximal_greedy(s, sums):
        """Add minimal elements to make s maximal using greedy blocking heuristic."""
        s = list(s)
        sums = set(sums)
        s_set = set(s)
        
        while True:
            # Find all unblocked elements
            unblocked = []
            for x in range(1, N + 1):
                if x not in s_set:
                    blocked = (2 * x in sums)
                    if not blocked:
                        for a in s:
                            if x + a in sums:
                                blocked = True
                                break
                    if not blocked:
                        unblocked.append(x)
            
            if not unblocked:
                break
            
            # Pick element that blocks the most unblocked elements
            best_x = None
            best_score = -1
            for x in unblocked:
                # Score: how many unblocked elements would become blocked
                score = 0
                # 2x sum blocks elements y where 2y == 2x (only y=x, already in s_set or not)
                # x+a sums block elements y where y+a' in {x+a} for some a' in s
                # More precisely, after adding x, new sums are {2x} ∪ {x+a : a in s}
                new_sums = set()
                new_sums.add(2 * x)
                for a in s:
                    new_sums.add(x + a)
                
                for y in unblocked:
                    if y == x:
                        continue
                    # Check if y would be blocked by s ∪ {x}
                    if 2 * y in sums or 2 * y in new_sums:
                        score += 1
                    else:
                        for a in s:
                            if y + a in sums:
                                score += 1
                                break
                        else:
                            # Check if y+x in sums or new_sums
                            if y + x in sums or y + x in new_sums:
                                score += 1
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            # Verify we can add best_x without Sidon violation
            new_sums = {2 * best_x} | {best_x + a for a in s}
            if sums.isdisjoint(new_sums):
                s.append(best_x)
                s_set.add(best_x)
                sums.update(new_sums)
            else:
                # Shouldn't happen if unblocked, but safety check
                break
        
        return s, sums
    
    best_result = None
    best_size = float('inf')
    
    random.seed(12345)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Generate random Sidon set of random size
        target_size = random.randint(8, 18)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        s = []
        sums = set()
        for x in candidates:
            if len(s) >= target_size:
                break
            new_sums = {2 * x}
            for a in s:
                new_sums.add(x + a)
            if sums.isdisjoint(new_sums):
                s.append(x)
                sums.update(new_sums)
        
        if len(s) < 3:
            continue
        
        # Make maximal
        s_max, sums_max = make_maximal_greedy(s, sums)
        
        # Verify maximality
        s_set = set(s_max)
        is_max = True
        for x in range(1, N + 1):
            if x not in s_set:
                if 2 * x in sums_max:
                    continue
                for a in s_max:
                    if x + a in sums_max:
                        break
                else:
                    is_max = False
                    break
        
        if is_max:
            size = len(s_max)
            if size < best_size:
                best_size = size
                best_result = s_max
                if size <= 11:
                    break
    
    if best_result is None:
        # Fallback: ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    return best_result
# EVOLVE-BLOCK-END
