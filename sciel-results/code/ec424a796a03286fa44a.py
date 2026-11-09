import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n = 16
    
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check using backtracking with MRV heuristic."""
        color = [-1] * n
        deg = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # MRV: pick uncolored vertex with most colored neighbors
            best_v = -1
            best_count = -1
            for v in range(n):
                if color[v] == -1:
                    cnt = sum(1 for u in range(n) if adj_mask[v] & (1 << u) and color[u] >= 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
            
            if best_v == -1:
                return True
            
            v = best_v
            used = set()
            for u in range(n):
                if adj_mask[v] & (1 << u) and color[u] >= 0:
                    used.add(color[u])
            
            # Try colors in order of least constraining
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def is_valid(adj_mask):
        """Check triangle-free and not 3-colorable."""
        # Triangle-free check
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                    if common:
                        return False
        return not is_3_colorable(adj_mask)
    
    # Build initial Grötzsch graph (11 vertices, triangle-free, 4-chromatic)
    adj_mask = [0] * n
    
    # Grötzsch: 5-cycle + Mycielski construction
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        add_edge(adj_mask, u, v)
    
    # Mycielski: add vertices 5-9 as copies, 10 as apex
    for u, w in cycle:
        add_edge(adj_mask, u, 5 + w)
        add_edge(adj_mask, w, 5 + u)
    for u in range(5):
        add_edge(adj_mask, 5 + u, 10)
    
    # Initialize best
    best_adj = list(adj_mask)
    best_count = edge_count(best_adj)
    
    # Phase 1: Greedy edge addition with constraint checking
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (adj_mask[i] & (1 << j))]
    
    def try_add_edges(adj_mask, edges_list, max_time):
        """Greedy: add edges that preserve validity."""
        changed = True
        while changed and time.time() < max_time:
            changed = False
            random.shuffle(edges_list)
            for u, v in edges_list:
                if time.time() > max_time:
                    break
                if adj_mask[u] & (1 << v):
                    continue
                # Check triangle-free
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                # Check 3-colorability
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                # Valid addition
                add_edge(adj_mask, u, v)
                changed = True
        
        return adj_mask
    
    # Run greedy addition
    adj_mask = try_add_edges(adj_mask, all_possible_edges, deadline - 30)
    current_count = edge_count(adj_mask)
    if current_count > best_count:
        best_count = current_count
        best_adj = list(adj_mask)
    
    # Phase 2: Local search with edge swaps (remove one, add another)
    existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                      if adj_mask[i] & (1 << j)]
    
    for iteration in range(500):
        if time.time() > deadline - 15:
            break
        
        # Randomly pick an edge to remove
        if not existing_edges:
            break
        ui, vi = random.choice(existing_edges)
        
        # Remove it
        remove_edge(adj_mask, ui, vi)
        
        # Try to add different edges
        candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                     if not (adj_mask[i] & (1 << j)) and (i,j) != (ui, vi)]
        random.shuffle(candidates)
        
        added_any = False
        for u, v in candidates:
            if time.time() > deadline - 15:
                break
            # Check triangle-free
            common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
            if common:
                continue
            # Check 3-colorability
            new_adj = list(adj_mask)
            add_edge(new_adj, u, v)
            if is_3_colorable(new_adj):
                continue
            add_edge(adj_mask, u, v)
            added_any = True
            break
        
        # If we couldn't add anything, try more candidates
        if not added_any:
            for u, v in candidates:
                if time.time() > deadline - 15:
                    break
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(adj_mask, u, v)
                added_any = True
                break
        
        current_count = edge_count(adj_mask)
        if current_count > best_count:
            best_count = current_count
            best_adj = list(adj_mask)
            # Update existing edges list
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                            if best_adj[i] & (1 << j)]
    
    # Phase 3: Simulated annealing style - accept worse moves sometimes
    current_adj = list(best_adj)
    current_count = best_count
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Pick random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add an edge
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (current_adj[i] & (1 << j))]
            if not candidates:
                continue
            u, v = random.choice(candidates)
            common = current_adj[u] & current_adj[v] & ~(1 << u) & ~(1 << v)
            if common:
                continue
            new_adj = list(current_adj)
            add_edge(new_adj, u, v)
            if is_3_colorable(new_adj):
                continue
            new_count = edge_count(new_adj)
            if new_count > current_count or random.random() < 0.1:
                current_adj = new_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = list(current_adj)
        
        elif move_type < 0.6:
            # Try to remove an edge
            existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if current_adj[i] & (1 << j)]
            if not existing:
                continue
            u, v = random.choice(existing)
            new_adj = list(current_adj)
            remove_edge(new_adj, u, v)
            if is_3_colorable(new_adj):
                continue  # Can't remove if it makes it 3-colorable
            new_count = edge_count(new_adj)
            # Only accept if it leads to improvement potential
            if new_count >= current_count - 2 and random.random() < 0.2:
                current_adj = new_adj
                current_count = new_count
        
        else:
            # Swap: remove one edge, try to add another
            existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if current_adj[i] & (1 << j)]
            if not existing:
                continue
            u1, v1 = random.choice(existing)
            new_adj = list(current_adj)
            remove_edge(new_adj, u1, v1)
            
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (new_adj[i] & (1 << j))]
            random.shuffle(candidates)
            
            found = False
            for u2, v2 in candidates[:20]:  # Limit search
                if time.time() > deadline - 5:
                    break
                common = new_adj[u2] & new_adj[v2] & ~(1 << u2) & ~(1 << v2)
                if common:
                    continue
                trial_adj = list(new_adj)
                add_edge(trial_adj, u2, v2)
                if is_3_colorable(trial_adj):
                    continue
                new_count = edge_count(trial_adj)
                if new_count > current_count or (new_count >= current_count - 1 and random.random() < 0.3):
                    current_adj = trial_adj
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = list(current_adj)
                    found = True
                    break
    
    # Final greedy cleanup
    final_adj = list(best_adj)
    candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                 if not (final_adj[i] & (1 << j))]
    random.shuffle(candidates)
    for u, v in candidates:
        if time.time() > deadline - 1:
            break
        common = final_adj[u] & final_adj[v] & ~(1 << u) & ~(1 << v)
        if common:
            continue
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if is_3_colorable(new_adj):
            continue
        add_edge(final_adj, u, v)
    
    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj
    
    # Return result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
