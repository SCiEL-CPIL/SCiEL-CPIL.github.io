import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 512
    elements = list(range(1, N + 1))
    
    # Precompute 4-APs for each element
    aps_for = {x: [] for x in elements}
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            aps_for[x1].append((x2, x3, x4))
            aps_for[x2].append((x1, x3, x4))
            aps_for[x3].append((x1, x2, x4))
            aps_for[x4].append((x1, x2, x3))
            
    def can_add(x, S):
        for a, b, c in aps_for[x]:
            if a in S and b in S and c in S:
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, deadline):
        S = set(S)
        while time.time() < deadline:
            # Try to add any available element
            # Optimization: iterate through a random permutation of candidates
            # to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            added = False
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
                    added = True
                    break
            
            if added:
                continue
            
            # Stuck. Try 1-exchange
            if not S:
                break
                
            improved = False
            rem_candidates = list(S)
            random.shuffle(rem_candidates)
            # Try up to 10 removals
            for y in rem_candidates[:10]:
                S.remove(y)
                # Try to add any element
                # Re-use candidates list? No, S changed.
                # Just iterate through all elements
                for z in range(1, N + 1):
                    if z not in S and z != y and can_add(z, S):
                        S.add(z)
                        improved = True
                        break
                if not improved:
                    S.add(y)
            
            if not improved:
                # If we failed to improve with 10 random removals, 
                # we might be in a deep local minimum.
                # Break to allow a new greedy restart.
                break

        return S

    best_set = set()
    deadline = time.time() + 85
    
    while time.time() < deadline:
        order = random.sample(elements, N)
        S = greedy_construct(order)
        S = local_search(S, deadline)
        
        if len(S) > len(best_set):
            best_set = S

    return sorted(best_set)
# EVOLVE-BLOCK-END
