import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def verify_c4_free(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
        return True

    def build_polarity_graph(q):
        """Generate the polarity graph of PG(2, q).
        Points are equivalence classes of [x:y:z] in F_q^3 \\ {0}.
        Two points are adjacent iff their dot product is 0 (orthogonal).
        For q odd, this is a C4-free graph."""
        points = []
        # Points with x != 0: represent as [1: a: b] for a,b in F_q
        for a in range(q):
            for b in range(q):
                points.append((1, a, b))
        # Points with x == 0, y != 0: represent as [0: 1: b] for b in F_q
        for b in range(q):
            points.append((0, 1, b))
        # Point with x == 0, y == 0: [0: 0: 1]
        points.append((0, 0, 1))
        
        n_pts = len(points)
        adj = np.zeros((n_pts, n_pts), dtype=bool)
        
        for i in range(n_pts):
            p1 = points[i]
            for j in range(i + 1, n_pts):
                p2 = points[j]
                dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                if dot == 0:
                    adj[i][j] = adj[j][i] = True
                    
        return adj

    def find_subgraph(adj, n_target, min_deg_target, seed=0, max_attempts=200):
        """Find an induced subgraph of size n_target with min degree >= min_deg_target."""
        n_pts = adj.shape[0]
        if n_pts < n_target:
            return None
            
        rng = random.Random(seed)
        removals_needed = n_pts - n_target
        
        best_subgraph = None
        best_min_deg = -1
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
                
            # Random perturbation: start with a random subset of size n_target + extra
            # Actually, let's try a different strategy:
            # 1. Start with all vertices
            # 2. Remove vertices using a randomized greedy that prioritizes low-degree vertices
            #    but with random tie-breaking and occasional "restarts"
            
            active_list = list(range(n_pts))
            current_degrees = np.zeros(n_pts, dtype=np.int32)
            
            for u in active_list:
                current_degrees[u] = np.sum(adj[u])
            
            # Remove vertices one by one
            for _ in range(removals_needed):
                if time.time() > deadline - 2:
                    break
                
                # Among active vertices, find those with minimum degree
                min_deg = n_pts + 1
                candidates = []
                for u in active_list:
                    d = current_degrees[u]
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                
                # Randomly pick from candidates
                v_to_remove = rng.choice(candidates)
                active_list.remove(v_to_remove)
                
                # Update degrees of remaining active vertices
                for u in active_list:
                    if adj[u, v_to_remove]:
                        current_degrees[u] -= 1
                
                current_degrees[v_to_remove] = -1  # Mark as removed
            
            if len(active_list) < n_target:
                continue
                
            min_deg_in_subgraph = min(current_degrees[u] for u in active_list)
            
            if min_deg_in_subgraph > best_min_deg:
                best_min_deg = min_deg_in_subgraph
                best_subgraph = active_list.copy()
                
            if best_min_deg >= min_deg_target:
                break
                
        if best_subgraph is not None and best_min_deg >= min_deg_target:
            sorted_subgraph = sorted(best_subgraph)
            if len(sorted_subgraph) != n_target:
                return None
                
            idx_map = {old: new for new, old in enumerate(sorted_subgraph)}
            edges = []
            for i in range(n_target):
                old_i = sorted_subgraph[i]
                for j in range(i + 1, n_target):
                    old_j = sorted_subgraph[j]
                    if adj[old_i, old_j]:
                        edges.append([i, j])
                        
            if verify_c4_free(n_target, edges):
                return {'n': n_target, 'edges': edges}
                
        return None

    def local_search_improve(adj, subgraph, n_target, min_deg_target, seed=0, max_iters=1000):
        """Try to improve a subgraph via local search (swap operations)."""
        rng = random.Random(seed)
        active = set(subgraph)
        current_min_deg = min(np.sum(adj[list(active)[i], :][list(active)]) for i in range(len(subgraph)))
        
        if current_min_deg >= min_deg_target:
            sorted_subgraph = sorted(active)
            edges = []
            for i in range(len(sorted_subgraph)):
                old_i = sorted_subgraph[i]
                for j in range(i + 1, len(sorted_subgraph)):
                    old_j = sorted_subgraph[j]
                    if adj[old_i, old_j]:
                        edges.append([i, j])
            if verify_c4_free(len(sorted_subgraph), edges):
                return {'n': len(sorted_subgraph), 'edges': edges}
        
        return None

    best_result = None
    
    # Strategy: Try polarity graphs of various sizes
    # For q=11: PG(2,11) has 133 points, degree ~12
    # For q=7: PG(2,7) has 57 points, degree ~8
    # For q=13: PG(2,13) has 183 points, degree ~14
    # For q=5: PG(2,5) has 31 points, degree ~6
    
    # We need min degree >= n - 39
    # For n=47: need min degree >= 8
    # For n=46: need min degree >= 7
    # For n=45: need min degree >= 6
    
    # q=11 polarity graph has 133 vertices, each of degree ~12
    # We can find an induced subgraph of size 47 with min degree >= 8
    
    adj_11 = build_polarity_graph(11)
    
    # Try to find n=47 first
    for n in [47, 46]:
        min_deg_req = n - 39
        if time.time() > deadline - 25:
            break
        res = find_subgraph(adj_11, n, min_deg_req, seed=42, max_attempts=100)
        if res:
            best_result = res
            break
            
    if not best_result:
        # Try q=7 (57 vertices)
        adj_7 = build_polarity_graph(7)
        for n in [46, 45]:
            min_deg_req = n - 39
            if time.time() > deadline - 15:
                break
            res = find_subgraph(adj_7, n, min_deg_req, seed=42, max_attempts=100)
            if res:
                best_result = res
                break
                
    if not best_result:
        # Try q=13 (183 vertices)
        adj_13 = build_polarity_graph(13)
        for n in [47, 46]:
            min_deg_req = n - 39
            if time.time() > deadline - 10:
                break
            res = find_subgraph(adj_13, n, min_deg_req, seed=42, max_attempts=50)
            if res:
                best_result = res
                break

    if not best_result:
        # Fallback: construct a simple C4-free graph with sufficient min degree
        # For small n, we can use known constructions
        # For n=39, empty graph has min degree 0 >= 0
        best_result = {'n': 39, 'edges': []}
        
    return best_result
