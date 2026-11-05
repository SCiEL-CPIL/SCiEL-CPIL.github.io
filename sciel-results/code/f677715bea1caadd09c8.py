import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_set = None
    best_size = float('inf')
    
    # Number of restarts to try
    # We can run until deadline or a fixed number of restarts
    # Let's try 1000 restarts, breaking early if time runs out.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        current_S = []
        current_sums = set()
        
        while True:
            # Find valid candidates
            valid_candidates = []
            # Optimization: We can skip checking x if it's already "blocked" in a way that prevents addition?
            # No, "blocked" for maximality is different from "invalid" for Sidon.
            # We just need Sidon validity.
            
            for x in range(1, N + 1):
                if x in current_S:
                    continue
                # Check Sidon
                # Sums involving x: 2x, x+a for a in S
                # Must be distinct from each other and from current_sums
                new_sums = [2*x]
                for a in current_S:
                    new_sums.append(x+a)
                    
                # Check distinctness within new_sums
                # 2x != x+a => x != a (true since x not in S)
                # x+a != x+b => a != b (true since S is distinct)
                # So new_sums are always distinct internally.
                
                # Check intersection with current_sums
                if not current_sums.intersection(new_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                # S is maximal
                break
                
            # Score each candidate
            best_x = None
            best_score = -1
            
            # To add some randomness and explore different local minima,
            # we can slightly perturb the score or pick randomly among top k.
            # But pure greedy blocking is a strong heuristic.
            
            for x in valid_candidates:
                new_sums_list = [2*x] + [x+a for a in current_S]
                
                # Identify y's that become newly blocked
                candidates_to_check = set()
                for s in new_sums_list:
                    if s % 2 == 0:
                        y = s // 2
                        if 1 <= y <= N and y not in current_S and y != x:
                            candidates_to_check.add(y)
                    for a in current_S:
                        y = s - a
                        if 1 <= y <= N and y not in current_S and y != x:
                            candidates_to_check.add(y)
                
                score = 0
                for y in candidates_to_check:
                    # Check if y was blocked before
                    if 2*y in current_sums:
                        continue
                    blocked_before = False
                    for a in current_S:
                        if y + a in current_sums:
                            blocked_before = True
                            break
                    if not blocked_before:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    
            if best_x is None:
                break
                
            # Update state
            current_S.append(best_x)
            current_sums.update([2*best_x] + [best_x+a for a in current_S[:-1]])
            
            if len(current_S) < best_size:
                best_size = len(current_S)
                best_set = list(current_S)
                
    if best_set is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
        
    return best_set
