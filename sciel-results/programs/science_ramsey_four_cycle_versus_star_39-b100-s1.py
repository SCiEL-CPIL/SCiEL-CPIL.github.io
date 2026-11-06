import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Polarity Graph of PG(2, q)
    # Points are lines in F_q^3 up to scalar.
    # There are q^2 + q + 1 points.
    # Two points are adjacent if they are orthogonal.
    # For q=7, n=57, degree=8.
    
    def get_polarity_graph(q):
        points = []
        # Generate all non-zero vectors, normalize
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero component 1
                    if x != 0:
                        inv_x = pow(x, -1, q)
                        points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                    elif y != 0:
                        inv_y = pow(y, -1, q)
                        points.append((0, 1, (z * inv_y) % q))
                    else:
                        inv_z = pow(z, -1, q)
                        points.append((0, 0, 1))
        
        # Remove duplicates
        unique_points = list(set(points))
        unique_points.sort()
        n_pts = len(unique_points)
        
        adj = np.zeros((n_pts, n_pts), dtype=np.int8)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                u = unique_points[i]
                v = unique_points[j]
                dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
                if dot == 0:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj, n_pts

    def try_polarity_subset(n_target, q=7):
        adj_full, n_full = get_polarity_graph(q)
        if n_full < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        # We want to remove 'num_remove' vertices such that
        # for every remaining vertex v, deg_full(v) - deg_removed(v) >= target_deg
        # deg_full(v) is 8 (for q=7, degree is q+1=8? or q=7? 
        # In PG(2, q), each point is on q+1 lines. The polarity graph connects orthogonal points.
        # The number of orthogonal points to a given point is q+1 (including itself if absolute?).
        # Usually polarity graphs are (q+1)-regular if no absolute points, or q-regular?
        # Let's check the degree in the generated graph.
        
        degs_full = adj_full.sum(axis=1)
        min_deg_full = np.min(degs_full)
        
        # Greedy removal
        # We need to select a set R of size num_remove.
        # Constraint: for all v in S, count(neighbors of v in R) <= deg_full(v) - target_deg.
        # Let limit[v] = deg_full(v) - target_deg.
        # We need a set R such that for all v not in R, |N(v) \cap R| <= limit[v].
        
        # Heuristic:
        # Start with R = empty.
        # Candidates for removal: all vertices.
        # We want to remove vertices that "cost" little to their neighbors.
        # Actually, we just need to find ANY valid R.
        
        # Backtracking search might be needed if greedy fails.
        # But let's try a greedy randomized approach.
        
        for attempt in range(500):
            if time.time() > deadline:
                return None
            
            R = set()
            # counts[v] = number of neighbors of v currently in R
            counts = np.zeros(n_full, dtype=np.int8)
            
            # Order of removal: random
            order = list(range(n_full))
            random.shuffle(order)
            
            removed = 0
            success = True
            
            # We need to remove exactly num_remove vertices.
            # But we must respect the limits.
            # If we pick a vertex v to remove, we increment counts[u] for all u in N(v).
            # We must ensure that for all u in N(v) \ R, counts[u] + 1 <= limit[u].
            # i.e., counts[u] < limit[u].
            
            # Wait, if u is in R, we don't care about its count?
            # No, the constraint is on S (remaining vertices).
            # So if u is in R, it's not in S, so no constraint on counts[u].
            # But if u is NOT in R, we need counts[u] <= limit[u].
            
            # So when considering removing v:
            # For each neighbor u of v:
            #   If u is in R: no problem.
            #   If u is NOT in R: we need counts[u] < limit[u].
            
            for v in order:
                if removed == num_remove:
                    break
                
                # Check if v can be removed
                can_remove = True
                for u in range(n_full):
                    if adj_full[v][u] == 1:
                        if u not in R:
                            if counts[u] >= degs_full[u] - target_deg:
                                can_remove = False
                                break
                
                if can_remove:
                    R.add(v)
                    removed += 1
                    for u in range(n_full):
                        if adj_full[v][u] == 1:
                            counts[u] += 1
            
            if removed == num_remove:
                # Verify
                S = [i for i in range(n_full) if i not in R]
                if len(S) != n_target:
                    continue
                
                # Build subgraph
                adj_sub = np.zeros((n_target, n_target), dtype=np.int8)
                old_to_new = {old: new for new, old in enumerate(S)}
                
                for new_i, old_i in enumerate(S):
                    for new_j, old_j in enumerate(S):
                        if new_i < new_j:
                            if adj_full[old_i][old_j]:
                                adj_sub[new_i][new_j] = 1
                                adj_sub[new_j][new_i] = 1
                
                degs_sub = adj_sub.sum(axis=1)
                if np.min(degs_sub) >= target_deg:
                    edges = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            if adj_sub[i][j]:
                                edges.append([i, j])
                    return {'n': n_target, 'edges': edges}
        
        return None

    # Try n=47, 46, 45...
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        # Try Polarity Graph Subset
        res = try_polarity_subset(n)
        if res:
            return res
        
        # Fallback: Random Greedy Construction
        # Start with empty graph, add edges greedily to maximize min degree.
        # This is less likely to succeed for high n, but worth a try.
        
        adj = np.zeros((n, n), dtype=np.int8)
        target_deg = n - 39
        
        # Initialize with a random matching or something?
        # Let's just do random edge addition.
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We want to maximize the minimum degree.
        # Greedy: add edge (u, v) if it doesn't create C4.
        # To bias towards high min degree, we can prioritize edges involving low-degree vertices.
        
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.shuffle(all_edges)
            adj = np.zeros((n, n), dtype=np.int8)
            degs = np.zeros(n, dtype=np.int8)
            
            for u, v in all_edges:
                # Check C4
                # Path u-w-x-v exists?
                # N_u = np.where(adj[u] == 1)[0]
                # For w in N_u: if np.any(adj[w] & adj[v]): return False
                
                # Optimization: use bitmasks
                # But n is small, numpy is fine.
                
                N_u = np.where(adj[u] == 1)[0]
                valid = True
                for w in N_u:
                    if np.any(adj[w] & adj[v]):
                        valid = False
                        break
                
                if valid:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    degs[u] += 1
                    degs[v] += 1
            
            if np.min(degs) >= target_deg:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # If nothing worked, return a trivial small graph
    # n=39, target=0. Empty graph works.
    return {'n': 39, 'edges': []}
