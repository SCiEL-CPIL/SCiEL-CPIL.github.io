"""Polarity graph on PG(2,5) with one extra edge, optimized via local search."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 50s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate all 31 points in PG(2,5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = pow(a, 3, 5)
                    vec = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    inv_b = pow(b, 3, 5)
                    vec = (0, 1, (c * inv_b) % 5)
                else:
                    inv_c = pow(c, 3, 5)
                    vec = (0, 0, 1)
                
                if vec not in seen:
                    seen.add(vec)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, x2, x3 = points[i]
            y1, y2, y3 = points[j]
            dot = (x1 * y1 + x2 * y2 + x3 * y3) % 5
            if dot == 0:
                adj[i][j] = adj[j][i] = True
                edge_set.add((i, j))
    
    # The polarity graph should have 90 edges.
    # We need exactly 91 edges, so we add one more edge.
    
    def count_common_neighbors(u, v, adj):
        count = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                count += 1
        return count
    
    # Find the best extra edge to add that minimizes C4 count
    best_extra = None
    best_c4_contrib = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edge_set:
                continue
            cn = count_common_neighbors(u, v, adj)
            c4_contrib = cn * (cn - 1) // 2
            if c4_contrib < best_c4_contrib:
                best_c4_contrib = c4_contrib
                best_extra = (u, v)
    
    if best_extra:
        u, v = best_extra
        adj[u][v] = adj[v][u] = True
        edge_set.add((u, v))
    
    def count_c4s(adj):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                cn = count_common_neighbors(u, v, adj)
                total += cn * (cn - 1) // 2
        return total // 2
    
    current_c4 = count_c4s(adj)
    
    best_adj = [row[:] for row in adj]
    best_c4_count = current_c4
    
    # Local search: try swapping edges
    # We use a simple hill-climbing approach with a time limit.
    # To speed up, we precompute the list of edges and non-edges.
    
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append((u, v))
        return edges
    
    current_adj = [row[:] for row in adj]
    current_edges = get_edge_list(current_adj)
    current_c4 = best_c4_count
    
    # Precompute non-edges list once, update as we go? 
    # For n=31, total pairs is 465. Edges are 91. Non-edges are 374.
    # Generating non-edges list is O(n^2) which is fast.
    
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 1000  # Allow more iterations since each is fast
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Pick a random edge to remove
        remove_e = random.choice(current_edges)
        u, v = remove_e
        
        # Pick a random non-edge to add
        # Efficiently find a non-edge
        # We can just iterate until we find one, or maintain a list.
        # For simplicity and speed, let's just pick random pairs until we find a non-edge.
        # Or better, maintain a list of non-edges.
        
        # Let's maintain a list of non-edges for O(1) selection.
        # This requires updating the list on every swap.
        # Initial non-edges:
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not current_adj[i][j]:
                    non_edges.append((i, j))
        
        if not non_edges:
            break
            
        add_e = random.choice(non_edges)
        
        # Perform swap
        # Remove (u,v)
        current_adj[u][v] = current_adj[v][u] = False
        # Add (u2, v2)
        u2, v2 = add_e
        current_adj[u2][v2] = current_adj[v2][u2] = True
        
        # Update edge lists
        # Remove (u,v) from current_edges, add (u2,v2)
        # Remove (u2,v2) from non_edges, add (u,v)
        # This is O(n) if we use lists.
        
        # To keep it simple and fast enough, we just recompute C4 count.
        # C4 count is O(n^3) ~ 30k ops. Very fast.
        
        new_c4 = count_c4s(current_adj)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            no_improve_count = 0
            if current_c4 < best_c4_count:
                best_c4_count = current_c4
                best_adj = [row[:] for row in current_adj]
                
            # Update edge/non-edge lists
            # Find and remove (u,v) from current_edges
            # Find and remove (u2,v2) from non_edges
            # Add (u2,v2) to current_edges
            # Add (u,v) to non_edges
            
            # For simplicity, we can just rebuild the lists if we accept a move.
            # Rebuilding is O(n^2) ~ 1000 ops. Acceptable.
            current_edges = get_edge_list(current_adj)
            # Rebuild non_edges
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not current_adj[i][j]:
                        non_edges.append((i, j))
        else:
            # Revert
            current_adj[u][v] = current_adj[v][u] = True
            current_adj[u2][v2] = current_adj[v2][u2] = False
            no_improve_count += 1
            
            # If we are stagnating, we might want to restart or break.
            # But with a good initial guess, we should be near optimal.
            
    best_edges = get_edge_list(best_adj)
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    result = [[u, v] for u, v in best_edges]
    return result
