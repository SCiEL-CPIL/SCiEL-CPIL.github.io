import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Build adjacency as list of bitmasks
    adj = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        if time.time() > deadline:
            return list(range(1, 10))
            
    # Reduce graph by removing dominated vertices
    # u is dominated by v if adj[u] & ~(1<<v) is subset of adj[v] & ~(1<<u)
    # i.e., N(u) \ {v} subset N(v) \ {u}
    # Since u and v are adjacent, N(u) includes v, N(v) includes u.
    # Condition: (adj[u] & ~(1<<v)) & ~adj[v] == 0
    
    active = [True] * (n + 1)
    for u in range(1, n + 1):
        if not active[u]:
            continue
        for v in range(1, n + 1):
            if u == v or not active[v]:
                continue
            # Check if u is dominated by v
            # They must be adjacent
            if not (adj[u] & (1 << v)):
                continue
            # N(u) - {v} must be subset of N(v) - {u}
            # N(u) - {v} = adj[u] & ~(1<<v)
            # N(v) - {u} = adj[v] & ~(1<<u)
            # Check if (adj[u] & ~(1<<v)) & ~(adj[v] & ~(1<<u)) == 0
            # Which is (adj[u] & ~(1<<v)) & ~adj[v] == 0
            # Note: ~(1<<u) is not needed in the check if we just check subset of adj[v]?
            # If x in N(u)\{v}, is x in N(v)?
            # If x == u, u is not in N(u)\{v}.
            # So we just check if all neighbors of u (except v) are neighbors of v.
            if (adj[u] & ~(1 << v) & ~adj[v]) == 0:
                active[u] = False
                break
                
    # Rebuild node list with only active nodes
    # Note: We can just filter in the search, or rebuild adj.
    # Rebuilding adj is cleaner for the bitmask logic if we re-index, 
    # but re-indexing 512 nodes is fast.
    # However, the current code uses 1-based indices up to 512.
    # If we remove nodes, the bitmasks still have holes, which is fine.
    
    def verify_solution(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                g = gcd(lst[i], lst[j])
                l = (lst[i] // g) * lst[j]
                if l > n:
                    return False
        return True
    
    # Popcount helper
    def popcount(x):
        return bin(x).count('1')
        
    # Efficient Tomita maximum clique with bitmasks
    def max_clique_tomita(adj, n, deadline, active):
        best_mask = [0]
        best_size = [0]
        
        def color_sort(P):
            """Greedy coloring of P, returns (ordered_list, color_list, max_color)."""
            if P == 0:
                return [], [], 0
            
            # Extract vertices
            P_list = []
            temp = P
            while temp:
                v = (temp & -temp).bit_length() - 1
                if active[v]:
                    P_list.append(v)
                temp &= temp - 1
            
            if not P_list:
                return [], [], 0

            # Order by degree in P (descending)
            P_sorted = sorted(P_list, key=lambda v: -popcount(adj[v] & P))
            
            # Greedy coloring: assign smallest color where no neighbor in same class
            color = {}
            color_classes = []  # list of (mask, [vertices])
            for v in P_sorted:
                placed = False
                for ci, (cmask, _) in enumerate(color_classes):
                    # v can go in class ci if no neighbor of v is in cmask
                    if (adj[v] & cmask) == 0:
                        color_classes[ci] = (cmask | (1 << v), color_classes[ci][1] + [v])
                        color[v] = ci + 1
                        placed = True
                        break
                if not placed:
                    color_classes.append((1 << v, [v]))
                    color[v] = len(color_classes)
            
            max_color = len(color_classes)
            # Order by color descending
            ordered = sorted(P_sorted, key=lambda v: -color[v])
            color_nums = [color[v] for v in ordered]
            return ordered, color_nums, max_color
        
        def expand(R_mask, R_size, P, X):
            if time.time() > deadline:
                return
            
            if R_size > best_size[0]:
                best_size[0] = R_size
                best_mask[0] = R_mask
            
            if P == 0:
                return
            
            ordered, color_nums, max_color = color_sort(P)
            
            if R_size + max_color <= best_size[0]:
                return
            
            # Process in reverse order (highest color first)
            for idx in range(len(ordered) - 1, -1, -1):
                v = ordered[idx]
                c = color_nums[idx]
                
                if R_size + c <= best_size[0]:
                    break
                
                if not (P & (1 << v)):
                    continue
                
                new_R_mask = R_mask | (1 << v)
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R_mask, R_size + 1, new_P, new_X)
                
                P &= ~(1 << v)
                X |= (1 << v)
                
                if time.time() > deadline:
                    return
        
        # Initial P: all active vertices
        P = 0
        for i in range(1, n + 1):
            if active[i]:
                P |= (1 << i)
                
        expand(0, 0, P, 0)
        
        # Extract result
        result = []
        temp = best_mask[0]
        while temp:
            v = (temp & -temp).bit_length() - 1
            result.append(v)
            temp &= temp - 1
        return result

    # Get a good initial solution via greedy (multiple orderings)
    def greedy_clique(order):
        clique = []
        clique_mask = 0
        for v in order:
            if not active[v]:
                continue
            if not clique:
                clique.append(v)
                clique_mask |= (1 << v)
            else:
                # v must be adjacent to all in clique: adj[v] must contain all bits in clique_mask
                if (adj[v] & clique_mask) == clique_mask:
                    clique.append(v)
                    clique_mask |= (1 << v)
        return clique

    # Try several orderings
    best_initial = []
    
    # Ordering 1: by degree ascending (most constrained first)
    active_nodes = [v for v in nodes if active[v]]
    order1 = sorted(active_nodes, key=lambda v: popcount(adj[v]))
    c1 = greedy_clique(order1)
    if len(c1) > len(best_initial):
        best_initial = c1
        
    # Ordering 2: by degree descending
    order2 = sorted(active_nodes, key=lambda v: -popcount(adj[v]))
    c2 = greedy_clique(order2)
    if len(c2) > len(best_initial):
        best_initial = c2
        
    # Ordering 3: by value
    c3 = greedy_clique(active_nodes)
    if len(c3) > len(best_initial):
        best_initial = c3
        
    # Ordering 4: by odd part, then by value
    def odd_part(x):
        while x % 2 == 0:
            x //= 2
        return x
    order4 = sorted(active_nodes, key=lambda v: (odd_part(v), v))
    c4 = greedy_clique(order4)
    if len(c4) > len(best_initial):
        best_initial = c4
        
    # Ordering 5: reverse by odd part
    order5 = sorted(active_nodes, key=lambda v: (-odd_part(v), v))
    c5 = greedy_clique(order5)
    if len(c5) > len(best_initial):
        best_initial = c5
        
    # Use best initial as lower bound for Tomita
    initial_size = len(best_initial)
    
    # Run Tomita with the initial solution as lower bound
    try:
        # Set initial best
        initial_mask = 0
        for v in best_initial:
            initial_mask |= (1 << v)
        
        # Modify Tomita to start with this lower bound
        best_mask = [initial_mask]
        best_size = [initial_size]
        
        def expand(R_mask, R_size, P, X):
            if time.time() > deadline:
                return
            
            if R_size > best_size[0]:
                best_size[0] = R_size
                best_mask[0] = R_mask
            
            if P == 0:
                return
            
            ordered, color_nums, max_color = color_sort(P)
            
            if R_size + max_color <= best_size[0]:
                return
            
            for idx in range(len(ordered) - 1, -1, -1):
                v = ordered[idx]
                c = color_nums[idx]
                
                if R_size + c <= best_size[0]:
                    break
                
                if not (P & (1 << v)):
                    continue
                
                new_R_mask = R_mask | (1 << v)
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R_mask, R_size + 1, new_P, new_X)
                
                P &= ~(1 << v)
                X |= (1 << v)
                
                if time.time() > deadline:
                    return
        
        P = 0
        for i in range(1, n + 1):
            if active[i]:
                P |= (1 << i)
                
        expand(0, 0, P, 0)
        
        result = []
        temp = best_mask[0]
        while temp:
            v = (temp & -temp).bit_length() - 1
            result.append(v)
            temp &= temp - 1
        best_clique = result
    except Exception:
        best_clique = list(best_initial)
    
    if not verify_solution(best_clique):
        best_clique = list(best_initial)
    
    # Local search: add vertices greedily
    improved = list(best_clique)
    improved_mask = 0
    for v in improved:
        improved_mask |= (1 << v)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in range(1, n + 1):
            if not active[v]:
                continue
            if improved_mask & (1 << v):
                continue
            if (adj[v] & improved_mask) == improved_mask:
                improved.append(v)
                improved_mask |= (1 << v)
                changed = True
    
    # 1-remove-2-add local search
    for _ in range(10):
        if time.time() > deadline - 3:
            break
        found_improvement = False
        for i in range(len(improved)):
            if time.time() > deadline - 3:
                break
            saved = improved.pop(i)
            saved_mask = improved_mask & ~(1 << saved)
            # Try to add 2+ vertices
            added = []
            added_mask = 0
            for v in range(1, n + 1):
                if not active[v]:
                    continue
                if (saved_mask | (1 << v)) & (1 << v) and v == saved:
                    continue
                if improved_mask & (1 << v):
                    continue
                if (adj[v] & (saved_mask | added_mask)) == (saved_mask | added_mask):
                    added.append(v)
                    added_mask |= (1 << v)
                if len(added) >= 2:
                    break
            if len(added) >= 2:
                improved.extend(added)
                improved_mask |= added_mask
                found_improvement = True
            else:
                improved.insert(i, saved)
                improved_mask |= (1 << saved)
        if not found_improvement:
            break
    
    # Final verification
    if not verify_solution(improved):
        if verify_solution(best_clique):
            return best_clique
        return best_initial
    
    return improved
