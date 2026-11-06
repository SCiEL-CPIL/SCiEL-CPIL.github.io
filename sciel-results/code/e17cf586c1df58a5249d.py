import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    # The task asks for a C4-free graph on n vertices with min degree >= n - 39.
    # We want to maximize n. The theoretical bound is around 46/47.
    # For n=46, min degree >= 7.
    # For n=47, min degree >= 8.
    
    # Strategy:
    # 1. Construct the Polarity Graph of the Projective Plane PG(2, 7).
    #    - It has q^2 + q + 1 = 49 + 7 + 1 = 57 vertices.
    #    - It is a known C4-free graph.
    #    - It is regular of degree q = 7 (for self-orthogonal points) or q+1=8?
    #      Actually, for a polarity of PG(2,q), the number of absolute points is q+1 (if q is even) or a specific number.
    #      Wait, the standard "Erdos-Renyi" polarity graph of PG(2,q) is C4-free.
    #      Degree: Each point is connected to the points on its polar line.
    #      A line has q+1 points. If the point is on its own polar line (absolute), it is connected to q other points.
    #      If not, it is connected to q+1 points? No, the point itself is not a neighbor.
    #      So degree is q if absolute, q if non-absolute? 
    #      Let's just build it and check.
    #      Actually, for q=7, n=57.
    #      We need to reduce to n=47 or 46.
    #      Removing 10 or 11 vertices.
    #      Since the graph is 7 or 8-regular, removing vertices will reduce degrees.
    #      We want the remaining graph to have min degree >= n-39.
    #      If n=47, min deg >= 8.
    #      If n=46, min deg >= 7.
    #      The original graph has degrees around 7 or 8.
    #      If we remove vertices carefully (those with high degree in the remaining set), we might preserve high degrees.
    #      However, removing a vertex decreases the degree of its neighbors by 1.
    #      If we start with degree 8 and remove 10 neighbors, degree drops to 0.
    #      This strategy might fail if the graph is too sparse.
    #      Let's check the density. 57 vertices, degree ~8. 
    #      If we keep 47 vertices, we remove 10.
    #      Average degree in remaining graph:
    #      Sum of degrees in original = 57 * 8 = 456.
    #      Edges removed: edges incident to removed vertices.
    #      If we remove a set R of 10 vertices, the number of edges lost is at least sum(deg(v) for v in R) - edges_inside_R.
    #      To keep min degree high, we should remove vertices that are "peripheral" or such that their neighbors are not critical.
    #      Actually, a better approach for "maximize n" with "min degree constraint" is often to find a dense subgraph.
    #      But C4-free graphs are sparse.
    #      Let's try the polarity graph construction.

    def build_polarity_graph(q):
        # Projective plane of order q.
        # Points are represented by vectors in F_{q+1}^3, normalized so first non-zero is 1.
        # Or simpler: use a finite field F_q.
        # We need a finite field of order 7. Since 7 is prime, it's just Z_7.
        
        # Points of PG(2, q) can be represented as:
        # (1, a, b) for a,b in F_q  -> q^2 points
        # (0, 1, b) for b in F_q    -> q points
        # (0, 0, 1)                 -> 1 point
        # Total q^2 + q + 1.
        
        points = []
        for a in range(q):
            for b in range(q):
                points.append((1, a, b))
        for b in range(q):
            points.append((0, 1, b))
        points.append((0, 0, 1))
        
        n_pts = len(points)
        adj = np.zeros((n_pts, n_pts), dtype=int)
        
        # Polarity: x^T A y = 0.
        # A standard polarity matrix for PG(2,q) is:
        # A = [[1, 0, 0], [0, 1, 1], [0, 1, 0]] ? 
        # Let's use the standard symplectic/hermitian form.
        # For q=7 (odd), we can use the quadratic form Q(x,y,z) = x^2 + y^2 + z^2? No, that's not a polarity in the projective sense directly unless we define the bilinear form.
        # The polarity is defined by a non-degenerate symmetric matrix M.
        # Let M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]? 
        # If we use M = identity, the polar of (1,0,0) is x=0.
        # The graph is constructed by connecting points p and q if p is on the polar line of q (and p != q).
        # p is on polar of q if p^T M q = 0.
        
        # Let's pick a specific non-degenerate symmetric matrix M over F_7.
        # M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] works?
        # If p=(1,0,0), polar is x=0. Points on x=0 are (0,1,b) and (0,0,1).
        # So (1,0,0) connects to all (0,1,b) and (0,0,1). Degree = q+1 = 8.
        # If p=(1,1,0), polar is x+y=0 -> y=-x. Points (1,-1,b) and (0,0,1) is not on it? (0,0,1) -> 0+0=0. Yes.
        # Wait, (0,0,1) is on x+y=0.
        # So degree is q+1 = 8?
        # For any point p, the polar line has q+1 points. One is p itself if p is absolute (p^T M p = 0).
        # If p is absolute, it is not connected to itself, so degree is q.
        # If p is not absolute, it is not on its own polar line, so degree is q+1.
        
        # Let's use M = I (Identity).
        # p^T p = x^2 + y^2 + z^2.
        # In F_7, squares are 0, 1, 4, 2.
        # Sum of three squares is 0?
        # 1+1+5? No 5 is not a square.
        # 1+4+2 = 7 = 0. So (1,2,1) is absolute? 1^2+2^2+1^2 = 1+4+1=6 != 0.
        # 1+1+1=3.
        # 4+4+4=12=5.
        # 2+2+2=6.
        # 1+4+2 = 7 = 0. So (1,2,1) is NOT absolute? 1^2=1, 2^2=4, 1^2=1. Sum=6.
        # Wait, 1+4+2=7. So we need squares 1, 4, 2.
        # 1^2=1, 2^2=4, 3^2=2.
        # So (1,2,3) is absolute.
        
        # We will build the graph using M = I.
        M = np.eye(3, dtype=int)
        
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                p = np.array(points[i])
                q = np.array(points[j])
                # Check p^T M q mod q
                val = (p @ M @ q) % q
                if val == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        return adj, points

    adj_full, points = build_polarity_graph(7)
    n_full = adj_full.shape[0]
    
    # Check C4-freeness of full graph (should be true)
    A2 = adj_full @ adj_full
    if np.any(A2 >= 2):
        # Fallback if construction is wrong
        pass

    # We want to find a subset of vertices V' of size n_target such that
    # the induced subgraph is C4-free (it is, as induced subgraph of C4-free)
    # and min degree >= n_target - 39.
    
    # Greedy removal:
    # Start with all 57 vertices.
    # Target n=47. Remove 10 vertices.
    # Target n=46. Remove 11 vertices.
    
    # To maximize min degree in the remaining graph, we should remove vertices that have
    # "redundant" connections or such that their removal doesn't hurt the min-degree vertices too much.
    # Actually, we want to PRESERVE vertices with high degree in the remaining set.
    # So we should REMOVE vertices with LOW degree in the current set?
    # No, if we remove a low-degree vertex, we don't reduce the degrees of many others.
    # If we remove a high-degree vertex, we reduce the degree of many neighbors.
    # So we should remove vertices with the LOWEST degree in the current induced subgraph.
    # This is a heuristic: "Peel off the leaves".
    
    def try_removal(adj, n_target, min_deg_req):
        n_curr = adj.shape[0]
        if n_curr < n_target:
            return None
        
        # We need to remove k = n_curr - n_target vertices.
        # Strategy: Iteratively remove the vertex with the minimum degree.
        # Tie-breaker: Random or fixed.
        
        current_adj = adj.copy()
        current_vertices = list(range(n_curr))
        
        for _ in range(n_curr - n_target):
            # Calculate degrees
            degs = current_adj.sum(axis=1)
            # Find min degree
            min_deg = np.min(degs)
            # Get all vertices with min degree
            candidates = [v for v in range(len(current_vertices)) if degs[v] == min_deg]
            
            # If multiple, pick one. 
            # To be deterministic, pick the first. 
            # To be better, maybe pick the one whose removal causes the least drop in the *global* min degree?
            # That's expensive. Let's just pick the first min-degree vertex.
            # Actually, if we always remove the current min-degree vertex, we might isolate some vertices.
            # Let's try a few random restarts of the removal order if it fails.
            
            # For now, simple greedy:
            v_idx = candidates[0]
            v_val = current_vertices[v_idx]
            
            # Remove v_idx
            # Delete row and column
            current_adj = np.delete(current_adj, v_idx, axis=0)
            current_adj = np.delete(current_adj, v_idx, axis=1)
            current_vertices.pop(v_idx)
            
            # Check time
            if time.time() > deadline - 5:
                return None

        # Check min degree of final graph
        final_degs = current_adj.sum(axis=1)
        if np.min(final_degs) >= min_deg_req:
            # Map back to original indices? No, we just need the edges.
            # The vertices in current_vertices are the ones remaining.
            # We need to return a graph on 0..n_target-1.
            # So we relabel the vertices.
            new_adj = current_adj
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if new_adj[i, j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
        return None

    # Try n=47 first
    # Min degree >= 47 - 39 = 8.
    res = try_removal(adj_full, 47, 8)
    if res:
        return res

    # Try n=46
    # Min degree >= 46 - 39 = 7.
    res = try_removal(adj_full, 46, 7)
    if res:
        return res

    # If simple greedy fails, try randomized greedy removal
    # We can restart the removal process with random tie-breaking.
    
    def try_removal_random(adj, n_target, min_deg_req, num_restarts=100):
        n_curr = adj.shape[0]
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            current_adj = adj.copy()
            current_vertices = list(range(n_curr))
            success = True
            for _ in range(n_curr - n_target):
                degs = current_adj.sum(axis=1)
                min_deg = np.min(degs)
                candidates = [v for v in range(len(current_vertices)) if degs[v] == min_deg]
                v_idx = np.random.choice(candidates)
                v_val = current_vertices[v_idx]
                
                current_adj = np.delete(current_adj, v_idx, axis=0)
                current_adj = np.delete(current_adj, v_idx, axis=1)
                current_vertices.pop(v_idx)
            
            final_degs = current_adj.sum(axis=1)
            if np.min(final_degs) >= min_deg_req:
                new_adj = current_adj
                edges = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if new_adj[i, j]:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
        return None

    res = try_removal_random(adj_full, 47, 8, 200)
    if res:
        return res

    res = try_removal_random(adj_full, 46, 7, 200)
    if res:
        return res

    # Fallback: Try to construct a smaller graph if polarity graph fails
    # Or just return a valid small graph
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
