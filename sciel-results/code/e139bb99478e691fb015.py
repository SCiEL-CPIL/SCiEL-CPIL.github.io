import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 192
    universe = list(range(1, n + 1))
    
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            
            # Check if any 3 distinct from others sum to multiple of d
            # We can use a set of residues to speed this up slightly, 
            # but L is small.
            # Target: u+v+w = k*d
            # u+v = k*d - w
            # Iterate w, then check if u+v exists.
            
            # Precompute pair sums of others?
            # L is at most 47. L^2/2 ~ 1100 pairs.
            # For each w, we need to know if target - w is a sum of two distinct others.
            
            # Let's just do triple loops for simplicity and speed in Python is okay for K<50
            # But 47^3 / 6 ~ 17k ops per d. 48 d's -> 800k ops.
            # This is fast enough.
            
            found = False
            for u_idx in range(L):
                u = others[u_idx]
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    # We need w distinct from u, v
                    # Sum so far u+v. We need w such that (u+v+w) % d == 0
                    # w % d == -(u+v) % d
                    # We can check if such w exists in others excluding u, v.
                    
                    # To avoid O(L) search for w, we can precompute counts of residues for 'others'
                    # But we need distinctness.
                    
                    # Let's stick to the triple loop but break early.
                    for w_idx in range(v_idx + 1, L):
                        w = others[w_idx]
                        if (u + v + w) % d == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if found:
                return False
        
        return True

    def can_add(S, x):
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            for u_idx in range(K):
                u = S_list[u_idx]
                for v_idx in range(u_idx + 1, K):
                    v = S_list[v_idx]
                    for w_idx in range(v_idx + 1, K):
                        w = S_list[w_idx]
                        if (u + v + w) % x == 0:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        # The triple is {x, u, v} where u, v in S \ {y} and u != v
        for y in S_list:
            # We need u, v from S such that u != y, v != y, u != v
            # And (x + u + v) % y == 0
            # u + v = -x (mod y)
            
            # Get candidates for u, v
            # If K < 2, no such pair exists (need 2 distinct from S, and if y is in S, 
            # we need 2 others. If K=1, others=0. If K=2, others=1. So need K>=3 to have 2 others)
            if K < 3:
                continue
            
            # Build list of elements in S excluding y
            # Actually, we can just iterate all pairs in S and check if they involve y
            # But it's faster to filter.
            others = [z for z in S_list if z != y]
            L = len(others)
            if L < 2:
                continue
            
            # Check if any pair in 'others' sums to -x mod y
            # We can use a set for residues?
            # L is small.
            for i in range(L):
                u = others[i]
                for j in range(i + 1, L):
                    v = others[j]
                    if (x + u + v) % y == 0:
                        return False
        
        return True

    best_S = set()
    best_size = 0
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        # Strategy 1: Greedy Descending
        # Strategy 2: Greedy Random
        # Strategy 3: Greedy Ascending (unlikely to be best, but try)
        
        # Let's do a greedy build
        S = set()
        
        # Randomize the order slightly to escape local optima
        # But descending is a very strong heuristic for this problem.
        # Let's mix: 80% descending, 20% random.
        if random.random() < 0.8:
            candidates = list(range(n, 0, -1))
        else:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        # Local Search
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try to add
            not_in_S = [x for x in range(1, n + 1) if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                for x in not_in_S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
            
            # If no add, try swap
            if not improved and S:
                # Pick a random element to remove
                r = random.choice(list(S))
                S.remove(r)
                
                # Try to add as many as possible
                not_in_S = [x for x in range(1, n + 1) if x not in S]
                random.shuffle(not_in_S)
                added = []
                for x in not_in_S:
                    if can_add(S, x):
                        S.add(x)
                        added.append(x)
                
                # If we added more than 1, or if we added 1 and the set is "better" somehow?
                # Actually, if we removed 1 and added 1, size is same. 
                # If we removed 1 and added >1, size increases.
                # If we removed 1 and added 0, size decreases.
                
                # We should only keep the change if size increased.
                # If size didn't increase, we might still keep it if it allows future growth?
                # For now, only accept if size increased.
                if len(S) > len(best_S) or (len(S) > len(S) - 1 - len(added) + 1): # This logic is messy
                    pass
                
                # Let's revert if size didn't increase
                # Original size before removal: len(S) + 1 - len(added)
                # Wait, we removed r, then added 'added'.
                # Size before phase: len(S) - len(added) + 1
                # Size after phase: len(S)
                # If len(S) > len(S) - len(added) + 1, i.e., len(added) > 1, then size increased.
                if len(added) <= 1:
                    # Revert
                    for x in added:
                        S.discard(x)
                    S.add(r)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
