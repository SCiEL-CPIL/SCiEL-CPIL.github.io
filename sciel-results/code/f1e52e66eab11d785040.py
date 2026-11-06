"""Graph on 16 vertices, 65 edges: minimize max triangle degree sum."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def compute_score(adj, deg, threshold=99):
        """Vectorized max triangle degree sum with early termination."""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = adj[u, v + 1:] & adj[v, v + 1:]
                    if np.any(common):
                        ws = np.where(common)[0] + v + 1
                        sums = deg[u] + deg[v] + deg[ws]
                        current_max = np.max(sums)
                        if current_max > max_sum:
                            max_sum = current_max
                            if max_sum >= threshold:
                                return max_sum
        return max_sum
    
    def init_k610_matching():
        """Initialize K_{6,10} + 5 matching edges in B with random labeling."""
        adj = np.zeros((n, n), dtype=np.int8)
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:6]
        B = vertices[6:]
        
        for u in A:
            for v in B:
                adj[u, v] = 1
                adj[v, u] = 1
        
        # 5 matching edges in B
        random.shuffle(B)
        for i in range(5):
            u, v = B[2 * i], B[2 * i + 1]
            adj[u, v] = 1
            adj[v, u] = 1
        
        deg = np.array([np.sum(adj[i]) for i in range(n)], dtype=np.int32)
        return adj, deg
    
    def simulated_annealing(adj, deg, time_limit, initial_temp=2.0, cooling=0.999):
        """Simulated annealing with degree tracking."""
        cur_adj = adj.copy()
        cur_deg = deg.copy()
        cur_score = compute_score(cur_adj, cur_deg)
        
        best_adj = cur_adj.copy()
        best_deg = cur_deg.copy()
        best_score = cur_score
        
        T = initial_temp
        steps = 0
        
        while time.time() < time_limit:
            # Pick random edge to remove
            attempts = 0
            removed = None
            while removed is None and attempts < 30:
                u, v = random.choice(all_pairs)
                if cur_adj[u, v]:
                    removed = (u, v)
                attempts += 1
            
            if removed is None:
                continue
            
            # Pick random non-edge to add
            attempts = 0
            added = None
            while added is None and attempts < 30:
                x, y = random.choice(all_pairs)
                if not cur_adj[x, y]:
                    added = (x, y)
                attempts += 1
            
            if added is None:
                continue
            
            u, v = removed
            x, y = added
            
            # Apply move
            cur_adj[u, v] = 0
            cur_adj[v, u] = 0
            cur_adj[x, y] = 1
            cur_adj[y, x] = 1
            cur_deg[u] -= 1
            cur_deg[v] -= 1
            cur_deg[x] += 1
            cur_deg[y] += 1
            
            new_score = compute_score(cur_adj, cur_deg)
            steps += 1
            
            if new_score < cur_score:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = cur_adj.copy()
                    best_deg = cur_deg.copy()
                    if best_score <= 23:
                        break
            elif new_score == cur_score:
                pass  # Accept equal
            else:
                delta = new_score - cur_score
                if random.random() < np.exp(-delta / max(T, 0.01)):
                    pass  # Accept worse
                else:
                    # Revert
                    cur_adj[u, v] = 1
                    cur_adj[v, u] = 1
                    cur_adj[x, y] = 0
                    cur_adj[y, x] = 0
                    cur_deg[u] += 1
                    cur_deg[v] += 1
                    cur_deg[x] -= 1
                    cur_deg[y] -= 1
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_deg, best_score
    
    def targeted_local_search(adj, deg, time_limit):
        """Local search that targets high-degree-sum triangles."""
        cur_adj = adj.copy()
        cur_deg = deg.copy()
        cur_score = compute_score(cur_adj, cur_deg)
        
        best_adj = cur_adj.copy()
        best_deg = cur_deg.copy()
        best_score = cur_score
        
        while time.time() < time_limit:
            if cur_score <= 23:
                break
            
            # Find the worst triangle
            worst_triangle = None
            worst_sum = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if cur_adj[u, v]:
                        common = cur_adj[u, v + 1:] & cur_adj[v, v + 1:]
                        if np.any(common):
                            ws = np.where(common)[0] + v + 1
                            for w in ws:
                                s = cur_deg[u] + cur_deg[v] + cur_deg[w]
                                if s > worst_sum:
                                    worst_sum = s
                                    worst_triangle = (u, v, w)
            
            if worst_triangle is None:
                break
            
            u, v, w = worst_triangle
            
            # Try to break this triangle by removing one of its edges
            # and adding a random non-edge elsewhere
            edges_to_try = [(u, v), (u, w), (v, w)]
            random.shuffle(edges_to_try)
            
            improved = False
            for ru, rv in edges_to_try:
                if time.time() > time_limit:
                    break
                
                # Find a random non-edge to add
                attempts = 0
                added = None
                while added is None and attempts < 50:
                    x, y = random.choice(all_pairs)
                    if not cur_adj[x, y]:
                        added = (x, y)
                    attempts += 1
                
                if added is None:
                    continue
                
                x, y = added
                
                # Apply
                cur_adj[ru, rv] = 0
                cur_adj[rv, ru] = 0
                cur_adj[x, y] = 1
                cur_adj[y, x] = 1
                cur_deg[ru] -= 1
                cur_deg[rv] -= 1
                cur_deg[x] += 1
                cur_deg[y] += 1
                
                new_score = compute_score(cur_adj, cur_deg)
                
                if new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = cur_adj.copy()
                        best_deg = cur_deg.copy()
                        if best_score <= 23:
                            return best_adj, best_deg, best_score
                    improved = True
                    break
                else:
                    # Revert
                    cur_adj[ru, rv] = 1
                    cur_adj[rv, ru] = 1
                    cur_adj[x, y] = 0
                    cur_adj[y, x] = 0
                    cur_deg[ru] += 1
                    cur_deg[rv] += 1
                    cur_deg[x] -= 1
                    cur_deg[y] -= 1
            
            if not improved:
                # Try a random swap
                attempts = 0
                removed = None
                while removed is None and attempts < 30:
                    u2, v2 = random.choice(all_pairs)
                    if cur_adj[u2, v2]:
                        removed = (u2, v2)
                    attempts += 1
                
                if removed is None:
                    continue
                
                u2, v2 = removed
                attempts = 0
                added = None
                while added is None and attempts < 30:
                    x, y = random.choice(all_pairs)
                    if not cur_adj[x, y]:
                        added = (x, y)
                    attempts += 1
                
                if added is None:
                    continue
                
                x, y = added
                cur_adj[u2, v2] = 0
                cur_adj[v2, u2] = 0
                cur_adj[x, y] = 1
                cur_adj[y, x] = 1
                cur_deg[u2] -= 1
                cur_deg[v2] -= 1
                cur_deg[x] += 1
                cur_deg[y] += 1
                
                new_score = compute_score(cur_adj, cur_deg)
                
                if new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = cur_adj.copy()
                        best_deg = cur_deg.copy()
                        if best_score <= 23:
                            return best_adj, best_deg, best_score
                else:
                    cur_adj[u2, v2] = 1
                    cur_adj[v2, u2] = 1
                    cur_adj[x, y] = 0
                    cur_adj[y, x] = 0
                    cur_deg[u2] += 1
                    cur_deg[v2] += 1
                    cur_deg[x] -= 1
                    cur_deg[y] -= 1
        
        return best_adj, best_deg, best_score
    
    random.seed(42)
    best_adj = None
    best_deg = None
    best_score = 999
    
    # Phase 1: Multiple restarts with simulated annealing
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        adj, deg = init_k610_matching()
        score = compute_score(adj, deg)
        
        if score < best_score:
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_score = score
        
        if best_score <= 23:
            break
        
        # Run SA with time budget
        sa_time_limit = deadline - 15.0 - restart * 5.0
        if sa_time_limit < start_time + 5.0:
            break
        
        adj2, deg2, score2 = simulated_annealing(adj, deg, sa_time_limit)
        if score2 < best_score:
            best_adj = adj2.copy()
            best_deg = deg2.copy()
            best_score = score2
        
        if best_score <= 23:
            break
    
    # Phase 2: Targeted local search on best
    if best_score > 23 and time.time() < deadline - 5.0:
        adj3, deg3, score3 = targeted_local_search(best_adj, best_deg, deadline - 3.0)
        if score3 < best_score:
            best_adj = adj3.copy()
            best_deg = deg3.copy()
            best_score = score3
    
    # Phase 3: Final polish with more SA iterations if still > 23
    if best_score > 23 and time.time() < deadline - 3.0:
        adj4, deg4, score4 = simulated_annealing(best_adj, best_deg, deadline - 1.0, initial_temp=1.0)
        if score4 < best_score:
            best_adj = adj4.copy()
            best_deg = deg4.copy()
            best_score = score4
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
    
    # Verify
    if len(final_edges) != target_edges:
        # Fallback: K_{6,10} + 5 matching
        final_edges = []
        for u in range(6):
            for v in range(6, 16):
                final_edges.append([u, v])
        for i in range(5):
            final_edges.append([6 + 2 * i, 7 + 2 * i])
    
    return {'n': n, 'edges': final_edges}
