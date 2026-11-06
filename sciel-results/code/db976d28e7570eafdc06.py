import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Seed solution
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best_result = {'n': seed_n, 'colors': seed_colors}
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 10:
            break
            
        # Time budget
        budget = 100.0 if n == 17 else 10.0
        target_end = time.time() + budget
        
        res = _sa_search(n, target_end)
        if res:
            return res
            
    return best_result

def _sa_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # colors array
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Adjacency matrices for each color
    # A[c] is n x n
    A = [np.zeros((n, n), dtype=int) for _ in range(4)]
    
    def update_matrices():
        for c in range(4):
            A[c] = np.zeros((n, n), dtype=int)
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            A[c][u, v] = 1
            A[c][v, u] = 1

    update_matrices()
    
    def compute_cost():
        cost = 0
        for c in range(4):
            # Compute A^5
            A5 = A[c].copy()
            for _ in range(4):
                A5 = A5 @ A[c]
            # For each edge (u, v) in color c, add (A5[u, v])
            # But A5 includes walks that might use (u, v).
            # We need walks in G \ {(u, v)}.
            # As discussed, this is tricky.
            # However, for a heuristic, using (A^5)[u, v] is okay.
            # But it will be non-zero even if no C6 exists (due to non-simple walks).
            # So cost will be > 0 even for valid solutions.
            # This makes SA hard to converge to 0.
            
            # Better: Just count the number of C6s exactly?
            # No, too slow for every step.
            
            # Let's stick to the walk count as a proxy.
            # If we find a state with low walk count, we check it exactly.
            for idx, (u, v) in enumerate(edges):
                if colors[idx] == c:
                    cost += A5[u, v]
        return cost

    def is_valid_exact():
        # Check for C6 in each color
        for c in range(4):
            # Build adjacency list for color c
            adj_list = [[] for _ in range(n)]
            for idx, (u, v) in enumerate(edges):
                if colors[idx] == c:
                    adj_list[u].append(v)
                    adj_list[v].append(u)
            
            # Check for C6
            # For each edge (u, v) in color c
            for idx, (u, v) in enumerate(edges):
                if colors[idx] != c:
                    continue
                
                # DFS from u to v, depth 5, avoiding (u, v)
                # Path: u -> x1 -> x2 -> x3 -> x4 -> v
                # x1 != v, x4 != u
                # All distinct
                
                def dfs(node, depth, visited):
                    if depth == 5:
                        return node == v
                    for nei in adj_list[node]:
                        if nei == v and depth == 4:
                            return True
                        if nei == u and depth < 4:
                            continue # Don't go back to start too early? 
                        # Actually, just check if nei is in visited
                        if nei in visited:
                            continue
                        if nei == v and depth < 4:
                            continue # Can't reach v early
                            
                        visited.add(nei)
                        if dfs(nei, depth + 1, visited):
                            return True
                        visited.remove(nei)
                    return False

                if dfs(u, 0, {u}):
                    return False
        return True

    current_cost = compute_cost()
    best_cost = current_cost
    best_colors = colors[:]
    
    T = 1.0
    cooling = 0.99
    
    # Check initial
    if is_valid_exact():
        return {'n': n, 'colors': colors}
        
    move_count = 0
    while time.time() < deadline:
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Pick new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Apply move
        colors[idx] = new_c
        A[old_c][u, v] = 0
        A[old_c][v, u] = 0
        A[new_c][u, v] = 1
        A[new_c][v, u] = 1
        
        new_cost = compute_cost()
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_colors = colors[:]
            
            # Check exact validity periodically
            if move_count % 1000 == 0:
                if is_valid_exact():
                    return {'n': n, 'colors': colors}
            
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
        else:
            # Revert
            colors[idx] = old_c
            A[old_c][u, v] = 1
            A[old_c][v, u] = 1
            A[new_c][u, v] = 0
            A[new_c][v, u] = 0
            new_cost = current_cost # Revert cost
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        move_count += 1
        
    # Final check on best found
    if best_cost < current_cost:
        colors = best_colors
        update_matrices() # Not strictly needed if we just return colors, but good for consistency
        if is_valid_exact():
            return {'n': n, 'colors': colors}
            
    return None
