import time
import random
from itertools import combinations

def _can_add(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    selected: list of currently selected integers.
    x: integer to add.
    
    Constraints to check:
    1. x does not divide the sum of any 3 distinct elements in selected.
    2. For each a in selected, a does not divide the sum of any 3 distinct elements 
       in (selected \ {a}) U {x}.
       Since selected was valid, a did not divide sums of 3 from selected \ {a}.
       We only need to check triples involving x.
       Triples involving x are {u, v, x} where u, v are distinct elements in selected \ {a}.
    """
    n = len(selected)
    if n == 0:
        return True
        
    # Check 1: x as divisor
    # Sum of any 3 distinct from selected
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = selected[i] + selected[j] + selected[k]
                if s % x == 0:
                    return False

    # Check 2: existing elements a as divisors
    # For each a in selected, check if a divides (u + v + x)
    # where u, v are distinct elements in selected \ {a}
    for idx_a, a in enumerate(selected):
        # Collect elements in selected excluding a
        # We can just iterate pairs and skip if index is idx_a
        for i in range(n):
            if i == idx_a:
                continue
            for j in range(i+1, n):
                if j == idx_a:
                    continue
                u = selected[i]
                v = selected[j]
                s = u + v + x
                if s % a == 0:
                    return False
                    
    return True

def _is_valid(selected):
    """Full validity check for debugging or final verification."""
    n = len(selected)
    for idx, a in enumerate(selected):
        others = [selected[i] for i in range(n) if i != idx]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _greedy(order, deadline):
    """Greedy construction based on given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """
    Improve the set by trying to add elements or swap elements.
    """
    best = list(selected)
    best_size = len(best)
    
    # Set for O(1) membership
    best_set = set(best)
    all_nums = set(range(1, 193))
    
    # We will iterate for a while or until no improvement
    # To avoid infinite loops on same state, we can randomize order
    
    max_rounds = 1000
    for r in range(max_rounds):
        if time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try adding a single element
        # Randomize the missing elements to explore different paths
        missing = list(all_nums - best_set)
        random.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                best_size = len(best)
                improved = True
                # After adding one, we can break to re-evaluate or continue?
                # Breaking allows us to try adding another in the next round or shuffle again.
                # Continuing might add more in same round. Let's continue to greedily add.
                # But shuffling was done once. 
                pass
        
        # 2. If size < 48, try swaps
        # Remove one, add one or two
        if len(best) < 48:
            # Shuffle best to try different removals
            indices = list(range(len(best)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                a = best[idx]
                # Temporarily remove a
                temp = best[:idx] + best[idx+1:]
                temp_set = best_set - {a}
                
                # Try to add elements not in temp
                missing = list(all_nums - temp_set)
                random.shuffle(missing)
                
                added = 0
                # Try to add up to 2 elements
                # We need to check validity incrementally
                # But _can_add takes a list. 
                # We can simulate adding to temp.
                
                # To be efficient, we can just try to add one first.
                # If we can add one, we have a swap of size 1 (net gain 0 if we removed 1 and added 1? 
                # No, we want to increase size. 
                # If we remove 1 and add 2, size increases by 1.
                # If we remove 1 and add 1, size stays same. This is only useful if it allows future additions.
                # Given the constraint, usually adding is the goal.
                
                # Let's try to add 2 elements to temp.
                # We need to check if x1 can be added to temp, then x2 to temp+x1.
                
                for x1 in missing:
                    if added >= 2:
                        break
                    if _can_add(x1, temp):
                        temp.append(x1)
                        temp_set.add(x1)
                        added += 1
                        # Try to add a second one
                        # Update missing? 
                        # Just iterate remaining missing
                        for x2 in missing:
                            if x2 == x1:
                                continue
                            if x2 in temp_set:
                                continue
                            if added >= 2:
                                break
                            if _can_add(x2, temp):
                                temp.append(x2)
                                temp_set.add(x2)
                                added += 1
                                break
                        
                        # If we added at least 1, and total size increased?
                        # Original size: L. Temp size: L-1+added.
                        # If added >= 1, size is L or L+1.
                        # We only want to accept if size increased OR if it's a "better" local move?
                        # For simplicity, only accept if size increased.
                        if len(temp) > len(best):
                            best = temp
                            best_set = temp_set
                            best_size = len(best)
                            improved = True
                            break # Restart swap search for this round? Or break inner loop?
                            # Breaking inner x1 loop.
                
                # If we broke due to improvement, we should probably restart the outer loop or continue carefully.
                # Let's just break out of the swap attempts for this round if we improved.
                if improved:
                    break
                    
        if not improved:
            # Try a more aggressive random restart? 
            # Or just break if no improvement in several rounds?
            # For now, break to avoid wasting time.
            break
            
    return best

def construct():
    start_time = time.time()
    # 60s limit. Use 55s deadline to be safe.
    deadline = start_time + 55.0
    
    # 1. Generate initial candidates using different greedy orders
    candidates = []
    
    # Descending (Large numbers first)
    order_desc = list(range(192, 0, -1))
    s1 = _greedy(order_desc, deadline)
    candidates.append(s1)
    
    # Ascending (Small numbers first)
    if time.time() < deadline:
        order_asc = list(range(1, 193))
        s2 = _greedy(order_asc, deadline)
        candidates.append(s2)
    
    # Randomized Greedy
    for seed in range(10):
        if time.time() > deadline - 5.0: # Leave 5s for local search
            break
        rng = random.Random(seed)
        order_rand = list(range(1, 193))
        rng.shuffle(order_rand)
        s = _greedy(order_rand, deadline)
        candidates.append(s)
        
    # Pick the best candidate
    best = max(candidates, key=len)
    
    # 2. Local Search
    if time.time() < deadline - 1.0:
        best = _local_search(best, deadline)
        
    # 3. Final Validation
    # _is_valid is O(N^4) which is slow for N=48 (approx 48*47*46*45/6 * 48 checks? No, combinations(47,3) * 48).
    # 48 * 16215 = ~780k checks. Fast enough.
    if not _is_valid(best):
        # Fallback to the best greedy solution that was valid
        # All greedy solutions are constructed to be valid.
        # So we can just pick the longest valid one from candidates.
        # But _local_search might have broken it? 
        # _local_search uses _can_add which is correct. 
        # So it should be valid. 
        # If not, fallback to s1 (descending greedy) which is definitely valid.
        best = s1
        
    return best
