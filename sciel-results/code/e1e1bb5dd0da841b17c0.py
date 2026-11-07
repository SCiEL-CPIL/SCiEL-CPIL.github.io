# EVOLVE-BLOCK-START
import time
import sys
import random

N = 128

def construct():
    # 1. Precompute forbidden hyperedges
    # A hyperedge is a set of 6 integers {a, b, c, a+b, a+c, b+c}
    # We forbid any subset of this set being fully present in our solution?
    # No, the rule is: "Forbid THREE DISTINCT members a,b,c for which a+b, a+c and b+c are also selected."
    # This means if {a,b,c} are in S, then at least one of {a+b, a+c, b+c} must NOT be in S.
    # Equivalently, S cannot contain {a, b, c, a+b, a+c, b+c}.
    # So the "forbidden masks" are exactly these sets of 6.
    
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
            
    # Precompute neighbors for dynamic ordering
    # neighbors[v] is a bitmask of all vertices u != v that share a hyperedge with v
    neighbors = [0] * (N + 1)
    for mask in forbidden_list:
        m = mask
        vertices_in_mask = []
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices_in_mask.append(v)
            m ^= lsb
        
        # For each pair in this hyperedge, they are neighbors
        # But we just need to know if they share *any* hyperedge.
        # So for each v in vertices_in_mask, add all other vertices in vertices_in_mask to neighbors[v]
        for i in range(len(vertices_in_mask)):
            v1 = vertices_in_mask[i]
            for j in range(len(vertices_in_mask)):
                if i != j:
                    v2 = vertices_in_mask[j]
                    neighbors[v1] |= (1 << v2)
    
    # 3. Greedy Initialization
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

    # Static ordering for initial greedy: High degree first
    static_order = list(range(1, N + 1))
    static_order.sort(key=lambda v: neighbors[v].bit_count(), reverse=True)
    
    best_size = 0
    best_mask = 0
    
    # Try a few random shuffles to find a good initial solution
    random.seed(42)
    for i in range(50):
        rand_order = static_order.copy()
        if i % 2 == 0:
            random.shuffle(rand_order)
        else:
            # Reverse high degree
            rand_order.sort(key=lambda v: neighbors[v].bit_count())
            
        S = greedy_init(rand_order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S

    # Also try the pure static order
    S = greedy_init(static_order)
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S

    # 4. Branch and Bound
    sys.setrecursionlimit(10000)
    start_time = time.time()
    timeout = 55
    
    # To speed up conflict checking, we can precompute a "conflict_signature" or just use the list.
    # The list check is O(degree). Degree is small.
    
    def can_add(v, selected_mask):
        for rest in conflicts[v]:
            if (selected_mask & rest) == rest:
                return False
        return True

    def search(selected_mask, available_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        # Pruning: Upper bound
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        # Time check
        if time.time() - start_time > timeout:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Dynamic Variable Ordering: Pick v in available with max (neighbors[v] & available).bit_count()
        # This is the "Most Constrained" variable.
        best_v = -1
        max_conflicts = -1
        
        m = available_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Count how many neighbors of v are also in available
            conf_count = (neighbors[v] & available_mask).bit_count()
            if conf_count > max_conflicts:
                max_conflicts = conf_count
                best_v = v
            m ^= lsb
            
        v = best_v
        lsb = 1 << v
        
        # Branch 1: Include v
        # Check if v can be added to selected_mask
        if can_add(v, selected_mask):
            new_selected = selected_mask | lsb
            # Update available: remove v, and remove any u in available that is now "blocked" by new_selected?
            # No, `available` in this formulation is just "vertices not decided yet".
            # The `can_add` check handles the constraint with `selected`.
            # However, we can prune `available` more aggressively.
            # If adding v makes some u impossible to add *later*?
            # The constraint is symmetric. If u is in available, it means u can be added to *current* selected.
            # After adding v, u might no longer be addable to `new_selected`.
            # So we should filter `available` to only those u that can be added to `new_selected`.
            # This is "Forward Checking".
            
            new_available = 0
            m = available_mask ^ lsb
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                if can_add(u, new_selected):
                    new_available |= lsb_u
                m ^= lsb_u
                
            search(new_selected, new_available, current_size + 1, depth + 1)
            
        # Branch 2: Exclude v
        # v is simply removed from available
        search(selected_mask, available_mask ^ lsb, current_size, depth + 1)

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
