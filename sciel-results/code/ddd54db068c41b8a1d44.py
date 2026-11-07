# EVOLVE-BLOCK-START
import time
import sys
import random

N = 128

def construct():
    # 1. Precompute forbidden hyperedges (sets of 6 integers)
    # A hyperedge H is invalid if H is a subset of S.
    # H = {a, b, c, a+b, a+c, b+c}
    # We need distinct a, b, c.
    # The sums a+b, a+c, b+c must be <= N.
    
    forbidden_masks = set()
    
    # a < b < c
    for c in range(1, N + 1):
        for b in range(1, c):
            if b + c > N:
                break
            for a in range(1, b):
                if a + c > N:
                    break
                # a+b is always <= a+c <= N
                h = {a, b, c, a + b, a + c, b + c}
                mask = 0
                for v in h:
                    mask |= (1 << v)
                forbidden_masks.add(mask)
                
    forbidden_list = list(forbidden_masks)
    
    # 2. Precompute conflicts for each vertex
    # conflicts[v] is a list of masks M such that if (selected & M) == M, 
    # then adding v would complete a forbidden set.
    # M = H \ {v}
    
    conflicts = [[] for _ in range(N + 1)]
    for mask in forbidden_list:
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            conflicts[v].append(rest)
            m ^= lsb
            
    # 3. Order vertices for B&B
    # Heuristic: Sort by number of conflicts (degree) descending.
    # Vertices with many constraints should be decided early.
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda v: len(conflicts[v]), reverse=True)
    
    sorted_vertices = vertices
    
    best_size = 0
    best_mask = 0
    
    # Greedy Initialization
    # Try adding vertices in sorted order (high degree first)
    def greedy_init(order):
        S = 0
        for v in order:
            can_add = True
            for rest in conflicts[v]:
                if (S & rest) == rest:
                    can_add = False
                    break
            if can_add:
                S |= (1 << v)
        return S

    # Local Search to improve greedy solution
    def local_search(S):
        improved = True
        while improved:
            improved = False
            # Try to add any vertex
            for u in range(1, N + 1):
                if not (S & (1 << u)):
                    can_add = True
                    for rest in conflicts[u]:
                        if (S & rest) == rest:
                            can_add = False
                            break
                    if can_add:
                        S |= (1 << u)
                        improved = True
                        break # Restart loop as S changed
        return S

    # Try a few orders
    # 1. Degree desc
    S1 = greedy_init(sorted_vertices)
    S1 = local_search(S1)
    best_size = S1.bit_count()
    best_mask = S1
    
    # 2. Degree asc
    S2 = greedy_init(sorted_vertices[::-1])
    S2 = local_search(S2)
    if S2.bit_count() > best_size:
        best_size = S2.bit_count()
        best_mask = S2
        
    # 3. Random shuffles
    random.seed(42)
    for _ in range(50):
        rand_order = sorted_vertices.copy()
        random.shuffle(rand_order)
        S = greedy_init(rand_order)
        S = local_search(S)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
            
    # 4. B&B Search
    # We use the sorted order to pick the next vertex.
    # This is a "fixed order" B&B.
    
    sys.setrecursionlimit(10000)
    start_time = time.time()
    
    def search(selected_mask, available_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        # Pruning: Upper bound
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        # Time check
        if time.time() - start_time > 55:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Find the next vertex to process
        # We want the vertex with the "most constraints" among available?
        # Or just the first one in our pre-sorted list?
        # Using the pre-sorted list ensures we process high-degree nodes first.
        # Let's find the first vertex in sorted_vertices that is in available_mask.
        
        v = -1
        for candidate in sorted_vertices:
            if available_mask & (1 << candidate):
                v = candidate
                break
                
        if v == -1:
            # Should not happen if available_mask != 0
            return
            
        lsb = 1 << v
        
        # Branch 1: Exclude v
        # This is often the better branch to try first if v is "expensive"?
        # Or include first?
        # If we include, we might prune more.
        # Let's try Exclude first, then Include.
        # Actually, if we Exclude, we just remove it.
        # If we Include, we check conflicts.
        
        # Try Exclude
        search(selected_mask, available_mask ^ lsb, current_size, depth + 1)
        
        # Try Include
        # Check if v can be added
        can_add = True
        for rest in conflicts[v]:
            if (selected_mask & rest) == rest:
                can_add = False
                break
                
        if can_add:
            new_selected = selected_mask | lsb
            new_available = available_mask ^ lsb
            search(new_selected, new_available, current_size + 1, depth + 1)
            
    # Run search
    # Initial available mask: all bits 1..N set
    init_available = 0
    for i in range(1, N + 1):
        init_available |= (1 << i)
        
    search(0, init_available, 0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
