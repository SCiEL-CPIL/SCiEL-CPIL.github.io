import time
import random

N = 512

# Precompute 4-APs
# ap[x] = list of tuples (a, b, c) such that {a, b, c, x} is a 4-AP
# To check if x can be added to S, we check if any (a,b,c) in ap[x] are all in S.
ap = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        if b > N: break
        c = b + d
        if c > N: break
        x = c + d
        if x > N: break
        # AP is a, b, c, x
        ap[x].append((a, b, c))
        # Also, x could be the first, second, or third term
        # But we are adding x to an existing set S.
        # The 4-AP is formed if the other 3 are in S.
        # So we just need to know the other 3 for every 4-AP containing x.
        
# Let's re-structure:
# For each x, list of (u, v, w) such that {u, v, w, x} is a 4-AP.
# We can generate all 4-APs and add to each member's list.
# This is O(N^2 * N) ? No, number of 4-APs is ~ N^2/12.
# Total entries in ap lists is 4 * (N^2/12) ~ N^2/3.
# N=512, N^2 ~ 262k. 262k/3 ~ 87k entries. Very manageable.

ap_conflicts = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        if b > N: break
        c = b + d
        if c > N: break
        e = c + d
        if e > N: break
        # AP: a, b, c, e
        ap_conflicts[a].append((b, c, e))
        ap_conflicts[b].append((a, c, e))
        ap_conflicts[c].append((a, b, e))
        ap_conflicts[e].append((a, b, c))

def can_add(S, x):
    """Check if x can be added to set S without forming a 4-AP."""
    for (u, v, w) in ap_conflicts[x]:
        if u in S and v in S and w in S:
            return False
    return True

def greedy_construct(order):
    S = set()
    for x in order:
        if can_add(S, x):
            S.add(x)
    return S

def local_search(S):
    S = set(S)
    improved = True
    while improved:
        improved = False
        # Phase 1: Add any valid element
        added = True
        while added:
            added = False
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    added = True
                    improved = True
                    break # restart loop to check others? or just continue?
                    # Restarting ensures we find all possible additions
                    # But it's O(N^2). N=512, N^2=262k. Fast.
        
        # Phase 2: Swap/Remove to allow more additions
        # Try removing each y in S
        for y in list(S):
            S.remove(y)
            # Try to add as many as possible
            temp_S = set(S)
            added_count = 0
            # Add any valid
            for x in range(1, N + 1):
                if x not in temp_S and can_add(temp_S, x):
                    temp_S.add(x)
                    added_count += 1
            
            if len(temp_S) > len(S) + 1: # We removed 1, so we need to add at least 2 to improve size
                # Actually, if len(temp_S) > len(S), it's an improvement.
                # Since we removed y, len(S) is current size.
                # If len(temp_S) > len(S), we gained.
                if len(temp_S) > len(S):
                    S = temp_S
                    improved = True
                    break # Restart local search with new S
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    best_S = set()
    best_size = 0
    
    # Initial Heuristic: Cantor Set (Base 3, digits 0,1)
    # This is a strong starting point.
    def is_cantor(x):
        while x > 0:
            if x % 3 == 1:
                return False
            x //= 3
        return True
        
    cantor_S = {x for x in range(1, N + 1) if is_cantor(x)}
    
    # Also try a simple greedy with random restarts
    # We will run multiple iterations of:
    # 1. Start with a seed (Cantor or Random Greedy)
    # 2. Local Search
    # 3. Update best
    
    # To diversify, we can shuffle the order for greedy initialization
    # or perturb the Cantor set.
    
    # Let's run a loop of local searches with different initializations
    
    # Initialization 1: Cantor + Greedy Add
    S1 = set(cantor_S)
    S1 = local_search(S1)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # Initialization 2: Random Greedy
    # We will do several random greedy runs
    for _ in range(20): # Adjust based on time
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S2 = greedy_construct(order)
        S2 = local_search(S2)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
            
    # Initialization 3: Perturbed Cantor
    # Randomly remove some elements from Cantor and re-optimize
    for _ in range(20):
        if time.time() > deadline:
            break
        S3 = set(cantor_S)
        # Remove 10% randomly
        to_remove = random.sample(list(S3), max(1, len(S3)//10))
        for x in to_remove:
            S3.remove(x)
        S3 = local_search(S3)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
            
    # Final Polish: Use remaining time for more local search passes on best_S
    # Or just return best_S
    
    if time.time() < deadline - 5:
        # Try to improve best_S further with more aggressive local search
        # e.g. trying to remove 2 elements
        pass
        
    return sorted(best_S)

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    
    N = 512
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Precompute conflicts
    ap_conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            if b > N: break
            c = b + d
            if c > N: break
            e = c + d
            if e > N: break
            ap_conflicts[a].append((b, c, e))
            ap_conflicts[b].append((a, c, e))
            ap_conflicts[c].append((a, b, e))
            ap_conflicts[e].append((a, b, c))

    def can_add(S, x):
        for (u, v, w) in ap_conflicts[x]:
            if u in S and v in S and w in S:
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Phase 1: Add any valid element
            added = True
            while added:
                added = False
                for x in range(1, N + 1):
                    if x not in S and can_add(S, x):
                        S.add(x)
                        added = True
                        improved = True
                        break
            
            # Phase 2: Remove one, add many
            for y in list(S):
                S.remove(y)
                temp_S = set(S)
                # Add any valid
                for x in range(1, N + 1):
                    if x not in temp_S and can_add(temp_S, x):
                        temp_S.add(x)
                
                if len(temp_S) > len(S):
                    S = temp_S
                    improved = True
                    break
        return S

    def is_cantor(x):
        while x > 0:
            if x % 3 == 1:
                return False
            x //= 3
        return True
        
    cantor_S = {x for x in range(1, N + 1) if is_cantor(x)}
    
    best_S = set()
    best_size = 0
    
    # 1. Cantor + Local Search
    S1 = set(cantor_S)
    S1 = local_search(S1)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # 2. Random Greedy + Local Search
    for _ in range(50): # Increase iterations if time permits
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S2 = greedy_construct(order)
        S2 = local_search(S2)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
            
    # 3. Perturbed Cantor + Local Search
    for _ in range(50):
        if time.time() > deadline:
            break
        S3 = set(cantor_S)
        # Remove 5-15% randomly
        k = random.randint(len(S3)//10, len(S3)//5)
        if k > 0:
            to_remove = random.sample(list(S3), k)
            for x in to_remove:
                S3.remove(x)
        S3 = local_search(S3)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)

    return sorted(best_S)
# EVOLVE-BLOCK-END
