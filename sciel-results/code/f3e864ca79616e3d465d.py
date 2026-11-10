import time
import random

def construct():
    # Deadline guard: 150s wall-clock limit, keep safety margin
    start_time = time.time()
    DEADLINE = 130.0  # 20s safety margin
    
    n = 16
    # Initialize adjacency matrix: 1 = edge present, 0 = edge absent
    # Start with complete graph K_16
    adj = [[1] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
    
    # Helper: check if graph is triangle-free
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    # Helper: exact backtracking 3-coloring check
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree (descending) for better pruning
        deg = [sum(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if color c is available for v
                conflict = False
                for u in range(n):
                    if adj[v][u] and colors[u] == c:
                        conflict = True
                        break
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Helper: count edges
    def count_edges(adj):
        return sum(sum(adj[i]) for i in range(n)) // 2
    
    # Helper: get list of edges
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # First, make the graph triangle-free by greedily removing edges
    # We'll use a randomized approach: for each triangle, remove one random edge
    def make_triangle_free(adj):
        adj = [row[:] for row in adj]
        # Repeatedly find a triangle and remove a random edge from it
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if changed:
                    break
                for j in range(i + 1, n):
                    if changed:
                        break
                    if not adj[i][j]:
                        continue
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            # Triangle (i, j, k)
                            # Remove one random edge from the triangle
                            choice = random.randint(0, 2)
                            if choice == 0:
                                adj[i][j] = 0
                                adj[j][i] = 0
                            elif choice == 1:
                                adj[i][k] = 0
                                adj[k][i] = 0
                            else:
                                adj[j][k] = 0
                                adj[k][j] = 0
                            changed = True
                            break
        return adj
    
    # Start with a random maximal triangle-free graph as initial solution
    best_adj = None
    best_edges = 0
    
    # Generate initial solution
    for attempt in range(10):
        if time.time() - start_time > DEADLINE:
            break
        # Random initial graph: keep each edge with probability 0.5
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        # Make triangle-free
        adj = make_triangle_free(adj)
        # Check if 4-chromatic
        if not is_3_colorable(adj):
            e = count_edges(adj)
            if e > best_edges:
                best_edges = e
                best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback: return Grötzsch graph
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
    
    # Simulated annealing to improve
    current_adj = [row[:] for row in best_adj]
    current_edges = best_edges
    best_adj = [row[:] for row in best_adj]
    best_edges = current_edges
    
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        # Choose a random pair (i, j)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Try to toggle edge (i, j)
        old_val = current_adj[i][j]
        current_adj[i][j] = 1 - old_val
        current_adj[j][i] = 1 - old_val
        
        # Check triangle-free constraint
        if not is_triangle_free(current_adj):
            # Revert
            current_adj[i][j] = old_val
            current_adj[j][i] = old_val
            T *= cooling
            continue
        
        new_edges = count_edges(current_adj)
        
        # Check 3-colorability
        if is_3_colorable(current_adj):
            # Not 4-chromatic, revert
            current_adj[i][j] = old_val
            current_adj[j][i] = old_val
            T *= cooling
            continue
        
        # Valid move
        delta = new_edges - current_edges
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            current_edges = new_edges
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [row[:] for row in current_adj]
        else:
            # Revert
            current_adj[i][j] = old_val
            current_adj[j][i] = old_val
        
        T *= cooling
    
    return get_edges(best_adj)
