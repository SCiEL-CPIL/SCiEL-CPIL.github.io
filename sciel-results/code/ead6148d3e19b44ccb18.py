import time
import random
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices with the maximum number of edges
    that is not 3-colorable.
    """
    n = 16
    deadline = time.time() + 115  # Safety margin
    
    # Initialize Clebsch Graph
    # Vertices are 0..15. Adjacent if Hamming distance is 2.
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if bin(u ^ v).count('1') == 2:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
    
    # Verify initial state
    # Clebsch is 40 edges, triangle-free, 4-chromatic (not 3-colorable)
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_3_colorable(adj):
        # Backtracking with bitsets
        # colors: 0, 1, 2
        # We use a recursive function.
        # To speed up, order vertices by degree descending.
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            avail = 7  # 111b
            
            # Check neighbors
            # Neighbors of v in the original graph
            # We need to check which colors are used by neighbors
            neighbors = adj[v]
            # Iterate through set bits in neighbors
            # But we need to map neighbor index to color
            # It's faster to just loop 0..n-1
            for u in range(n):
                if (neighbors >> u) & 1:
                    if colors[u] != -1:
                        avail &= ~(1 << colors[u])
            
            if avail == 0:
                return False
            
            # Try available colors
            # Heuristic: Try colors that appear most often? Or just 0,1,2
            for c in range(3):
                if (avail >> c) & 1:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)

    # Initial check
    if not is_triangle_free(adj):
        return [list(e) for e in edges]
    if is_3_colorable(adj):
        # Should not happen for Clebsch, but if it does, we need a fallback or just return it?
        # The task requires NOT 3-colorable.
        # If Clebsch is 3-colorable, we have a problem.
        # Literature says Clebsch is 4-chromatic.
        pass

    best_adj = adj[:]
    best_edges = edges[:]
    best_count = len(edges)
    
    current_adj = adj[:]
    current_edges = set(edges)
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for faster lookup?
    # Or just use the set.
    
    def add_edge(adj, u, v):
        # Returns True if successful (no triangle created)
        if (adj[u] >> v) & 1:
            return False # Already exists
        if adj[u] & adj[v]:
            return False # Triangle
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        if not ((adj[u] >> v) & 1):
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline:
        # Generate move
        # We try to increase edge count.
        # Move types:
        # 1. Add 1 edge (Net +1)
        # 2. Remove 1, Add 2 (Net +1)
        # 3. Remove 2, Add 3 (Net +1)
        
        move_type = random.random()
        
        # Prepare rollback
        old_adj = current_adj[:]
        old_edges = set(current_edges)
        
        success = False
        new_adj = current_adj[:]
        new_edges = set(current_edges)
        
        if move_type < 0.3:
            # Try to add 1 edge
            # Find a non-edge that doesn't create a triangle
            # In Clebsch, there are none. So this will likely fail initially.
            # But after swaps, there might be.
            candidates = []
            for u, v in all_possible_edges:
                if not ((new_adj[u] >> v) & 1):
                    if not (new_adj[u] & new_adj[v]):
                        candidates.append((u, v))
            if candidates:
                u, v = random.choice(candidates)
                if add_edge(new_adj, u, v):
                    new_edges.add((u, v))
                    success = True
        elif move_type < 0.6:
            # Remove 1, Add 2
            if len(current_edges) > 0:
                e_rem = random.choice(list(current_edges))
                u_rem, v_rem = e_rem
                # Remove it
                remove_edge(new_adj, u_rem, v_rem)
                new_edges.discard(e_rem)
                
                # Try to add 2 edges
                added = 0
                # Get candidates that don't create triangles with current state
                # Note: Adding edges might create triangles with each other?
                # If we add (u1, v1) and (u2, v2), they might form a triangle if they share a vertex and have a common neighbor?
                # No, triangle is 3 vertices.
                # If we add (a,b) and (c,d), a triangle could be formed if e.g. a=c and b,d have common neighbor?
                # Actually, adding edges one by one and checking triangle-freeness at each step is safe.
                
                # Candidates for first add
                cands1 = []
                for u, v in all_possible_edges:
                    if (u, v) not in new_edges and not ((u, v) in new_edges or (v, u) in new_edges):
                        # Check triangle
                        if not (new_adj[u] & new_adj[v]):
                            cands1.append((u, v))
                
                if cands1:
                    u1, v1 = random.choice(cands1)
                    if add_edge(new_adj, u1, v1):
                        new_edges.add((u1, v1))
                        added += 1
                        
                        # Candidates for second add
                        cands2 = []
                        for u, v in all_possible_edges:
                            if (u, v) not in new_edges and not ((u, v) in new_edges or (v, u) in new_edges):
                                if not (new_adj[u] & new_adj[v]):
                                    cands2.append((u, v))
                        
                        if cands2:
                            u2, v2 = random.choice(cands2)
                            if add_edge(new_adj, u2, v2):
                                new_edges.add((u2, v2))
                                added += 1
                
                if added == 2:
                    success = True
        
        else:
            # Remove 2, Add 3
            if len(current_edges) >= 2:
                e_rem1, e_rem2 = random.sample(list(current_edges), 2)
                u1, v1 = e_rem1
                u2, v2 = e_rem2
                
                remove_edge(new_adj, u1, v1)
                remove_edge(new_adj, u2, v2)
                new_edges.discard(e_rem1)
                new_edges.discard(e_rem2)
                
                added = 0
                for _ in range(3):
                    cands = []
                    for u, v in all_possible_edges:
                        if (u, v) not in new_edges:
                            if not (new_adj[u] & new_adj[v]):
                                cands.append((u, v))
                    if not cands:
                        break
                    u, v = random.choice(cands)
                    if add_edge(new_adj, u, v):
                        new_edges.add((u, v))
                        added += 1
                
                if added == 3:
                    success = True
        
        if not success:
            continue
            
        # Check if new state is valid
        # 1. Triangle free (guaranteed by construction)
        # 2. Not 3-colorable
        if is_3_colorable(new_adj):
            # Rollback
            current_adj = old_adj
            current_edges = old_edges
            continue
            
        # Calculate score
        # We want to maximize edges.
        # Accept if better or with probability
        delta = len(new_edges) - len(current_edges)
        
        # Energy = -edges
        # E_new = -len(new_edges)
        # E_curr = -len(current_edges)
        # delta_E = E_new - E_curr = -delta
        
        if delta > 0:
            # Always accept improvement
            current_adj = new_adj
            current_edges = new_edges
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_adj = current_adj[:]
                best_edges = list(current_edges)
                
        else:
            # delta <= 0
            # Accept with probability
            if random.random() < pow(2.71828, -(-delta) / T): # exp(delta/T)
                current_adj = new_adj
                current_edges = new_edges
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart or reheat?
            # For now, just keep going.
            
    return [list(e) for e in best_edges]

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
