import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 31
    target_edges = 91
    
    # --- 1. Structured Initialization: Polarity Graph of PG(2,5) ---
    
    def generate_polarity_graph():
        """
        Generates the polarity graph of the projective plane of order q=5.
        Vertices: Points of PG(2, q). There are q^2 + q + 1 = 31 points.
        Edges: Two points are connected if they are orthogonal w.r.t the polarity.
        """
        q = 5
        # Points of PG(2, q) can be represented as pairs (x, y) in F_q^2 and points at infinity.
        # Total points: q^2 + q + 1 = 25 + 5 + 1 = 31.
        
        edges = set()
        
        # Affine-Affine edges
        for i in range(25):
            x1, y1 = i % 5, i // 5
            for j in range(i+1, 25):
                x2, y2 = j % 5, j // 5
                dx = (x1 - x2) % 5
                dy = (y1 - y2) % 5
                if (dx*dx + dy*dy) % 5 == 0:
                    edges.add((i, j))
        
        # Affine-Infinity edges
        # Infinity is index 25.
        # (x,y) connected to inf if x^2 + y^2 = 0.
        for i in range(25):
            x, y = i % 5, i // 5
            if (x*x + y*y) % 5 == 0:
                edges.add((i, 25))
                
        return edges

    init_edges = generate_polarity_graph()
    
    # Adjust edge count to exactly 91
    if len(init_edges) > target_edges:
        edge_list = list(init_edges)
        random.shuffle(edge_list)
        while len(init_edges) > target_edges:
            e = edge_list.pop()
            init_edges.discard(e)
    elif len(init_edges) < target_edges:
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        non_edges = [e for e in all_edges if e not in init_edges]
        random.shuffle(non_edges)
        while len(init_edges) < target_edges and non_edges:
            init_edges.add(non_edges.pop())
            
    # --- 2. Optimized Local Search with Spectral Awareness ---
    
    # Build adjacency matrix for fast computation
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in init_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    def count_c4():
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.dot(adj[u], adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    current_score = count_c4()
    best_score = current_score
    best_edges = set(init_edges)
    
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Spectral helper: compute trace(A^4) which equals sum of eigenvalues^4
    # This is 4 * C4_count + 4 * (number of triangles) + ... but for our purposes,
    # minimizing trace(A^4) correlates with minimizing C4s.
    def spectral_cost():
        # trace(A^2) = 2*E (constant), trace(A^3) relates to triangles, trace(A^4) relates to C4s
        A2 = adj @ adj
        A4 = A2 @ A2
        return np.trace(A4)
    
    current_spectral = spectral_cost()
    best_spectral = current_spectral
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        current_edges = set(init_edges)
        non_edges = [e for e in all_edges_list if e not in current_edges]
        
        random.shuffle(non_edges)
        
        # Precompute neighbor sets for faster delta calculation
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    neighbors[u].add(v)
        
        for e_rm in list(current_edges):
            if (time.time() - start_time) > deadline - 5:
                break
                
            u_rm, v_rm = e_rm
            
            best_delta = 0
            best_e_add = None
            
            # Sample non-edges with spectral awareness
            sample_size = min(300, len(non_edges))
            sampled = random.sample(non_edges, sample_size) if len(non_edges) > sample_size else non_edges
            
            # Sort candidates by spectral potential: prefer adding edges between vertices
            # with high common neighbors (which would create many C4s if added) - wait, we want to AVOID that.
            # Actually, we want to add edges that create FEW C4s. So prefer pairs with low common neighbor count.
            def spectral_potential(e):
                u, v = e
                common = len(neighbors[u] & neighbors[v])
                return common
            
            # Sort by potential (lowest first - these add fewer C4s)
            sampled.sort(key=spectral_potential)
            
            # Take top candidates (lowest C4 creation) plus some random for diversity
            top_candidates = sampled[:100]
            random_candidates = random.sample(sampled, min(50, len(sampled)))
            candidates = list(set(top_candidates + random_candidates))
            
            for e_add in candidates:
                u_add, v_add = e_add
                
                # Quick check: if u_add==u_rm or v_add==u_rm etc, handle carefully
                # Calculate delta C4
                # Remove (u_rm, v_rm): decreases C4s that use this edge
                # Add (u_add, v_add): increases C4s that use this edge
                
                # For speed, compute approximate delta using common neighbors
                # C4s using edge (u,v) = C(common(u,v), 2) where common(u,v) = |N(u) ∩ N(v)|
                
                common_rm = len(neighbors[u_rm] & neighbors[v_rm])
                c4s_removed = common_rm * (common_rm - 1) // 2
                
                common_add = len(neighbors[u_add] & neighbors[v_add])
                # When we add (u_add, v_add), the new C4s are those using this edge
                # But some of those C4s might already be counted if the edge existed? No, it's a non-edge.
                # So new C4s = C(common_add, 2)
                # But wait, adding the edge also changes common neighbors for other pairs.
                # For a quick estimate, use just the direct contribution.
                c4s_added = common_add * (common_add - 1) // 2
                
                delta_est = c4s_added - c4s_removed
                
                if delta_est < best_delta:
                    # Verify with exact calculation
                    adj[u_rm, v_rm] = 0
                    adj[v_rm, u_rm] = 0
                    adj[u_add, v_add] = 1
                    adj[v_add, u_add] = 1
                    
                    new_score = count_c4()
                    delta = new_score - current_score
                    
                    adj[u_rm, v_rm] = 1
                    adj[v_rm, u_rm] = 1
                    adj[u_add, v_add] = 0
                    adj[v_add, u_add] = 0
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_e_add = e_add
                    
                    # Update neighbors after reverting (they didn't change since we reverted)
                    
            if best_e_add is not None and best_delta < 0:
                u_rm, v_rm = e_rm
                u_add, v_add = best_e_add
                init_edges.discard((u_rm, v_rm))
                init_edges.add((u_add, v_add))
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1
                
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(init_edges)
                improved = True
                break 

    return [[u, v] for u, v in best_edges]
