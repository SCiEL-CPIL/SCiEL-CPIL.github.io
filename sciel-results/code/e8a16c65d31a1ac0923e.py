import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    
    # Build conflict graph: conflict[a] = set of b values where (a+b) | a*b
    # (a+b) | a*b  <=>  (a+b) | a^2  (since a*b = a*(a+b) - a^2)
    # So for each a, find divisors d of a^2 with d > a, set b = d - a
    # Need 1 <= b <= N, b != a
    
    conflict = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find all divisors d of a_sq where d > a
        # d = a + b, b > 0
        # b = d - a, need 1 <= b <= N, b != a
        # So d ranges from a+1 to a+N (since b <= N means d <= a+N)
        # But also d must divide a_sq
        # Efficient: iterate d from a+1 to min(a_sq, a+N)
        upper = min(a_sq, a + N)
        for d in range(a + 1, upper + 1):
            if a_sq % d == 0:
                b = d - a
                if 1 <= b <= N and b != a:
                    conflict[a].add(b)
                    conflict[b].add(a)
    
    # Greedy independent set: start empty, add vertices in order of fewest conflicts
    # Then do local search with swaps
    
    def greedy_is():
        # Order vertices by degree (fewest conflicts first)
        degrees = [len(conflict[i]) for i in range(1, N + 1)]
        order = list(range(1, N + 1))
        order.sort(key=lambda x: degrees[x])
        
        selected = set()
        for v in order:
            # Check if v conflicts with any selected vertex
            can_add = True
            for u in conflict[v]:
                if u in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected
    
    best_set = greedy_is()
    best_size = len(best_set)
    
    # Local search: try swaps to improve
    # Also try adding/removing
    
    def is_valid_set(s):
        for a in s:
            for b in s:
                if a != b:
                    if b in conflict[a]:
                        return False
        return True
    
    # Local search with swaps
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        current_set = set(best_set)
        improved = False
        
        # Try single additions
        for v in range(1, N + 1):
            if v not in current_set:
                can_add = True
                for u in conflict[v]:
                    if u in current_set:
                        can_add = False
                        break
                if can_add:
                    current_set.add(v)
                    improved = True
                    break
        
        if improved:
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(best_set)
            continue
        
        # Try swaps: remove one, add one
        for v in list(current_set):
            for w in range(1, N + 1):
                if w in current_set:
                    continue
                # Remove v, can we add w?
                can_add_w = True
                for u in conflict[w]:
                    if u in current_set and u != v:
                        can_add_w = False
                        break
                if can_add_w:
                    # This swap keeps size same, but might enable future additions
                    # Only do it if it reduces total conflicts
                    old_conflicts = sum(1 for x in current_set for y in conflict[x] if y in current_set)
                    new_set = (current_set - {v}) | {w}
                    new_conflicts = sum(1 for x in new_set for y in conflict[x] if y in new_set)
                    if new_conflicts < old_conflicts:
                        current_set = new_set
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(best_set)
    
    # Final check
    return list(best_set)
