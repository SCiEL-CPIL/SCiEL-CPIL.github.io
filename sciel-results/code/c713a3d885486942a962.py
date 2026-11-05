import time
import math
import random
from functools import lru_cache

N = 128

# Precompute conflicts: conflicts[x] is a set of (u,v) pairs such that {x,u,v} is a forbidden triple
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    """
    For each triple a < b < c, check if lcm(a,b) == lcm(a,c) == lcm(b,c).
    If so, mark conflicts.
    """
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # c must be > b, <= N, and divide L_ab
            # Also lcm(a,c) == L_ab and lcm(b,c) == L_ab
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

# Precompute a "conflict graph" for local search:
# Two elements x, y are "conflicting" if there exists z such that {x,y,z} is forbidden.
# Actually, for local search, we need to check if adding x to a set S creates a forbidden triple.
# That happens if there exist u, v in S such that (x,u,v) is a forbidden triple.
# So the "conflict" of x with S depends on S.
# For a pair (x,y), they are in a forbidden triple together if there exists z such that (x,y,z) is forbidden.
# But that's not enough; we need to know if z is in S.

# For the "can_add" check, we need to check if for any (u,v) in conflicts[x], both u and v are in S.

def can_add(x, S):
    """Check if x can be added to set S without creating a forbidden triple."""
    for u, v in conflicts[x]:
        if u in S and v in S:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = set()
    best_size = 0
    
    # Strategy:
    # 1. Greedy construction with random restarts to get a good initial solution.
    # 2. Local search (add/remove/swap) to improve.
    # 3. Branch and Bound (B&B) with the greedy solution as a lower bound for pruning.
    
    # --- Phase 1: Greedy with restarts ---
    def greedy_once(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    # Try different orderings
    # 1. Natural order
    # 2. Reverse order
    # 3. Random orders
    
    best_greedy = set()
    
    # Natural
    S = greedy_once(range(1, N + 1))
    if len(S) > len(best_greedy):
        best_greedy = S
    
    # Reverse
    S = greedy_once(range(N, 0, -1))
    if len(S) > len(best_greedy):
        best_greedy = S
    
    # Random orders
    for _ in range(200):
        if time.time() > deadline - 20:  # Keep 20s for other phases
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_once(order)
        if len(S) > len(best_greedy):
            best_greedy = S
    
    best_set = best_greedy
    best_size = len(best_set)
    
    # --- Phase 2: Local Search ---
    # Try to add elements that are not in best_set but can be added.
    # Then try to remove elements to allow adding more.
    
    def local_search(S, max_iters=10000):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            
            # Try adding any element not in S
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break  # Restart scan after change
            
            if improved:
                continue
            
            # Try removing one element and then adding up to two?
            # Or just remove one and see if we can add more.
            # Let's try: for each y in S, remove y, then greedily add as many as possible.
            # Keep the best result.
            best_S_local = set(S)
            best_size_local = len(S)
            
            # Sample some removals to save time
            removals = list(S)
            random.shuffle(removals)
            for y in removals[:20]:  # Try removing up to 20 random elements
                if time.time() > deadline - 10:
                    break
                S_temp = set(S)
                S_temp.discard(y)
                # Greedily add
                for x in range(1, N + 1):
                    if x not in S_temp and can_add(x, S_temp):
                        S_temp.add(x)
                
                if len(S_temp) > best_size_local:
                    best_size_local = len(S_temp)
                    best_S_local = S_temp
            
            if len(best_S_local) > len(S):
                S = best_S_local
                improved = True
        
        return S
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # --- Phase 3: Branch and Bound ---
    # Use B&B to try to find a larger set.
    # We use the current best_size as the lower bound for pruning.
    # We branch on elements in a specific order.
    
    # To make B&B effective, we need a good ordering.
    # Let's order elements by their "degree" in the conflict graph (number of pairs in conflicts[x]).
    # High degree elements are more restrictive, so we should branch on them early?
    # Or low degree?
    # Usually, high degree first helps prune.
    
    degrees = [0] * (N + 1)
    for x in range(1, N + 1):
        degrees[x] = len(conflicts[x])
    
    # Sort elements by degree descending
    order = sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True)
    
    # B&B State:
    # current_set: set of chosen elements
    # remaining: list of elements not yet considered
    # 
    # Pruning:
    # if len(current_set) + len(remaining) <= best_size: return
    
    # We need to be careful with the "can_add" check in B&B.
    # When we include x, we must filter `remaining` to remove elements that conflict with x *given current_set*.
    # An element y conflicts with x (given current_set) if there exists z in current_set such that (x,y,z) is forbidden?
    # No, y conflicts with x if adding y to (current_set U {x}) would create a forbidden triple.
    # That happens if there exist u, v in (current_set U {x}) such that (y,u,v) is forbidden.
    # Since y is not in current_set, the pairs (u,v) involving x are (x, u) for u in current_set, or (x, x) which is not a pair of distinct elements.
    # So y conflicts with x if:
    # 1. There exist u, v in current_set such that (y,u,v) is forbidden. (y already conflicts with current_set)
    # 2. There exists u in current_set such that (y,x,u) is forbidden. (y conflicts with x relative to current_set)
    
    # Let's refine can_add for B&B.
    # can_add(y, S) checks if y can be added to S.
    
    def bnb_rec(current_set, remaining, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_set) + len(remaining) <= best_size:
            return
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Pick an element to branch on.
        # We pick the first element in `remaining`.
        x = remaining[0]
        rest = remaining[1:]
        
        # Branch 1: Include x
        new_set = current_set | {x}
        # Filter rest: keep y if can_add(y, new_set)
        valid_rest = []
        for y in rest:
            if can_add(y, new_set):
                valid_rest.append(y)
        
        bnb_rec(new_set, valid_rest, depth + 1)
        
        # Branch 2: Exclude x
        bnb_rec(current_set, rest, depth + 1)
    
    # Run B&B with a few different orderings
    # The order affects performance significantly.
    # Let's try the degree-sorted order first.
    bnb_rec(set(), order, 0)
    
    # If we have time, try a few random shuffles of the order
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            bnb_rec(set(), order, 0)
    
    # Final check
    if not best_set:
        best_set = {1}
    
    return list(best_set)
