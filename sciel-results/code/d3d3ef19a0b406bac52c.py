import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety
    
    N = 512
    
    # Precompute 4-APs
    # aps_by_element[x] will contain a list of tuples (i, j, k) 
    # where (x, i, j, k) form a 4-AP (in some order, but we store the OTHER 3)
    aps_by_element = [[] for _ in range(N)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            aps_by_element[a].append((b, c, e))
            aps_by_element[b].append((a, c, e))
            aps_by_element[c].append((a, b, e))
            aps_by_element[e].append((a, b, c))

    def can_add(x, S):
        for i, j, k in aps_by_element[x]:
            if S[i] and S[j] and S[k]:
                return False
        return True

    def get_blockers(x, S):
        # Returns a list of elements in S that are part of a "complete" AP with x
        # If x is blocked, there is at least one AP (x, i, j, k) where i,j,k are in S.
        # We return the union of all such i,j,k.
        blockers = set()
        for i, j, k in aps_by_element[x]:
            if S[i] and S[j] and S[k]:
                blockers.add(i)
                blockers.add(j)
                blockers.add(k)
        return list(blockers)

    def add_all(S, count):
        # Try to add all missing elements
        changed = True
        while changed:
            changed = False
            for x in range(N):
                if not S[x]:
                    if can_add(x, S):
                        S[x] = 1
                        count += 1
                        changed = True
        return S, count

    def greedy_init():
        S = bytearray(N)
        elements = list(range(N))
        random.shuffle(elements)
        count = 0
        for x in elements:
            if can_add(x, S):
                S[x] = 1
                count += 1
        return S, count

    def local_search(S, count):
        # 1. Add all possible
        S, count = add_all(S, count)
        
        # 2. Swap search
        # We try to swap elements to potentially allow adding more later
        # Or just improve the set structure.
        
        # Get blocked elements
        blocked = [x for x in range(N) if not S[x] and not can_add(x, S)]
        
        # We will perform a limited number of swap attempts
        # Each swap: remove y, add x.
        
        # To allow improvement, after a swap, we call add_all again.
        
        swap_attempts = 0
        max_swap_attempts = 100 # Limit per local search call
        
        while swap_attempts < max_swap_attempts:
            if not blocked:
                break
            
            # Pick a random blocked element
            x = random.choice(blocked)
            
            # Get its blockers
            blockers = get_blockers(x, S)
            
            if not blockers:
                # Should not happen if can_add returned False, but just in case
                blocked.remove(x)
                continue
                
            # Pick a random blocker to remove
            y = random.choice(blockers)
            
            # Try swap: remove y, add x
            S[y] = 0
            count -= 1
            
            if can_add(x, S):
                S[x] = 1
                count += 1
                # Swap successful. 
                # Now try to add more
                S, count = add_all(S, count)
                
                # Update blocked list? It's expensive to recalculate.
                # Just recalculate blocked list if we want to continue swapping
                # Or just restart local search?
                # Let's just recalculate blocked list for efficiency in this loop
                blocked = [x for x in range(N) if not S[x] and not can_add(x, S)]
            else:
                # Revert
                S[y] = 1
                count += 1
            
            swap_attempts += 1
            
            # Check time
            if time.time() - start_time > time_limit - 5.0:
                break
                
        return S, count

    best_count = 0
    best_S = bytearray(N)
    
    # Initial run
    S, count = greedy_init()
    S, count = local_search(S, count)
    
    if count > best_count:
        best_count = count
        best_S = S.copy()
        
    while time.time() - start_time < time_limit:
        # Perturb best
        S = best_S.copy()
        count = best_count
        
        # Remove 5-10 random elements to perturb
        k = random.randint(5, 10)
        current_elements = [i for i in range(N) if S[i]]
        random.shuffle(current_elements)
        for i in range(min(k, len(current_elements))):
            S[current_elements[i]] = 0
            count -= 1
            
        S, count = local_search(S, count)
        
        if count > best_count:
            best_count = count
            best_S = S.copy()
            
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if best_S[i]]
    
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
