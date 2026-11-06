import time
import random
import numpy as np

def _compute_max_book_fast2(adj_mask, n=40):
    """Compute max book size: for each edge, count common neighbors."""
    A = adj_mask.astype(np.int32)
    A2 = A @ A
    max_book = 0
    # Get all edges (upper triangle)
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i, j]:
                common = A2[i, j]
                if common > max_book:
                    max_book = common
    return max_book

def _compute_edge_books(adj_mask, n=40):
    """Compute book size for every edge. Returns dict {(u,v): book}."""
    A = adj_mask.astype(np.int32)
    A2 = A @ A
    books = {}
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i, j]:
                books[(i, j)] = A2[i, j]
    return books

def _deterministic_near_bipartite(n=40, target_edges=401):
    """
    Construct a near-bipartite graph: K_{20,20} minus k edges, plus k+1 intra-part edges.
    Total edges = 400 - k + (k+1) = 401.
    Choose k and the specific edges to minimize max book.
    
    Strategy: Remove k bipartite edges that are 'critical' (shared by many potential
    intra-edges) and add k+1 intra-edges spread to minimize their book sizes.
    """
    # Partition: A = {0..19}, B = {20..39}
    A_verts = list(range(20))
    B_verts = list(range(20, 40))
    
    best_adj = None
    best_max = 999
    
    # Try different values of k (number of removed bipartite edges)
    for k in range(1, 15):
        # We need to remove k bipartite edges and add k+1 intra edges
        # The intra edges' book size = common neighbors
        # For an intra-A edge (u,v): common neighbors = |{b in B : u~b and v~b}|
        # In full K_{20,20}, that's 20. After removing some bipartite edges, it decreases.
        
        # Greedy: for each intra edge we want to add, find the pair with fewest common neighbors
        # after accounting for removed bipartite edges.
        
        # Start with full bipartite
        adj = np.zeros((n, n), dtype=bool)
        for a in A_verts:
            for b in B_verts:
                adj[a, b] = True
                adj[b, a] = True
        
        # We'll iteratively: pick an intra edge, remove bipartite edges to reduce its book
        # Actually, let's think differently:
        # Book of intra-A edge (u,v) = number of b in B adjacent to both u and v
        # To minimize this, we want to remove bipartite edges (u,b) or (v,b) for many b
        
        # Strategy: Choose intra edges first, then remove bipartite edges to minimize their books
        # while keeping total edge count at 401.
        
        # For k+1 intra edges, we need to remove k bipartite edges.
        # Each removed bipartite edge (a,b) reduces the book of intra-A edges (a,v) and (u,a) by 1
        # (if v~b and u~b respectively).
        
        # Let's try: pick intra edges that are "far apart" and remove bipartite edges
        # that are shared by multiple intra-edge endpoints.
        
        # Simpler approach: enumerate small sets of intra edges
        if k + 1 <= 12:  # Limit enumeration
            # Try all combinations of k+1 intra-A edges from a reduced candidate set
            # To keep it tractable, only consider intra edges where vertices are "spread out"
            
            # Candidate intra-A edges: pairs (i, j) where j - i >= 3 (spread out)
            candidates = []
            for i in range(20):
                for j in range(i+3, 20):
                    candidates.append((i, j))
            
            # Also consider intra-B edges
            candidates_B = []
            for i in range(20, 40):
                for j in range(i+3, 40):
                    candidates_B.append((i, j))
            
            # For small k, try a few specific configurations
            # Config: place intra edges to minimize max common-neighbor count
            
            # For k+1 intra-A edges (u_i, v_i), book of (u_i, v_i) = 
            #   |{b in B : adj[u_i,b] and adj[v_i,b]}|
            # = 20 - |{b : not adj[u_i,b] or not adj[v_i,b]}|
            
            # If we remove bipartite edge (a, b), it reduces book of any intra-A edge (a, v) or (u, a)
            # that has b as common neighbor.
            
            # Greedy construction:
            # 1. Choose k+1 intra-A edges
            # 2. For each, compute initial book (20)
            # 3. Remove k bipartite edges to minimize the max book
            
            # Try a specific pattern: intra edges (0, 10), (1, 11), ..., (k, k+10)
            # These share no vertices, so removing bipartite edge (a, b) only helps edges involving a.
            
            intra_edges = []
            for idx in range(k + 1):
                u = idx
                v = idx + 10
                if v < 20:
                    intra_edges.append((u, v))
                else:
                    intra_edges.append((u, 10))  # fallback
            
            # Add intra edges to adj
            for u, v in intra_edges:
                adj[u, v] = True
                adj[v, u] = True
            
            # Now compute books and find which bipartite edges to remove
            # Book of intra edge (u,v) = common neighbors in B
            # We can remove up to k bipartite edges to reduce books
            
            # For each intra edge (u,v), its book = 20 initially
            # Removing (u, b) reduces book of (u,v) by 1
            # Removing (v, b) reduces book of (u,v) by 1
            # Removing (a, b) where a not in {u,v} doesn't help this edge
            
            # So for edge (u,v), we can reduce its book by removing edges incident to u or v in B
            # Each removed edge helps at most 2 intra edges (if it's incident to a vertex in two intra edges)
            
            # Greedy: repeatedly remove the bipartite edge that most reduces the max book
            current_books = {}
            for u, v in intra_edges:
                # Count common neighbors in B
                cn = 0
                for b in B_verts:
                    if adj[u, b] and adj[v, b]:
                        cn += 1
                current_books[(u, v)] = cn
            
            # Also need to check bipartite edge books
            # For bipartite edge (a, b) where a in A, b in B:
            # Common neighbors = {x in A : x~b and x~a} union {x in B : x~a and x~b}
            # = {x in A\{a} : x~b} (since no intra-B edges) union {x in B\{b} : x~a} (none)
            # Wait: x~a for x in B means x is in B and adjacent to a. In K_{20,20}, all B vertices are adjacent to a.
            # x~b for x in B: no intra-B edges, so only if x=b (but x!=b). So no B vertices are common neighbors.
            # x~a for x in A: only intra-A neighbors of a. x~b for x in A: all A vertices adjacent to b.
            # So common neighbors of (a,b) = {x in A : x~a and x~b} = {x in A\{a} : x~a} (since all x in A are ~b)
            # = intra-A neighbors of a (excluding a itself, but a isn't adjacent to itself)
            # = number of intra-A edges incident to a
            
            # So bipartite edge (a,b) has book = deg_intra_A(a)
            # Max book from bipartite edges = max over a in A of deg_intra_A(a)
            
            max_bip_book = 0
            for a in A_verts:
                deg = 0
                for u, v in intra_edges:
                    if a == u or a == v:
                        deg += 1
                if deg > max_bip_book:
                    max_bip_book = deg
            
            # Now greedily remove bipartite edges to reduce intra-edge books
            removed = set()
            for _ in range(k):
                # Find the bipartite edge whose removal most reduces max book
                best_gain = 0
                best_edge = None
                for u, v in intra_edges:
                    if current_books[(u, v)] <= max_bip_book:
                        continue
                    # Try removing (u, b) for some b
                    for b in B_verts:
                        if (u, b) in removed or (b, u) in removed:
                            continue
                        if not adj[u, b]:
                            continue
                        # Gain: reduces book of (u,v) by 1, and possibly other intra edges involving u
                        gain = 0
                        # How many intra edges involving u have b as common neighbor?
                        for u2, v2 in intra_edges:
                            if u == u2 or u == v2:
                                if adj[u2, b] and adj[v2, b]:
                                    gain += 1
                        # Also, removing (u,b) doesn't change bipartite books (deg_intra unchanged)
                        if gain > best_gain:
                            best_gain = gain
                            best_edge = (u, b)
                    # Try removing (v, b)
                    for b in B_verts:
                        if (v, b) in removed or (b, v) in removed:
                            continue
                        if not adj[v, b]:
                            continue
                        gain = 0
                        for u2, v2 in intra_edges:
                            if v == u2 or v == v2:
                                if adj[u2, b] and adj[v2, b]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_edge = (v, b)
                
                if best_edge is None or best_gain == 0:
                    break
                
                # Remove the edge
                u_r, b_r = best_edge
                adj[u_r, b_r] = False
                adj[b_r, u_r] = False
                removed.add((u_r, b_r))
                
                # Update books
                for u, v in intra_edges:
                    if (u_r == u or u_r == v) and adj[u, b_r] == False and adj[v, b_r]:
                        # Wait, need to check if b_r was a common neighbor
                        pass  # Recompute below
            
            # Recompute all books
            max_book = max_bip_book
            for u, v in intra_edges:
                cn = 0
                for b in B_verts:
                    if adj[u, b] and adj[v, b]:
                        cn += 1
                if cn > max_book:
                    max_book = cn
            
            # Check total edges
            edge_count = np.sum(adj) // 2
            if edge_count == target_edges and max_book < best_max:
                best_max = max_book
                best_adj = adj.copy()
        
        if best_max <= 7:  # Good enough, stop
            break
    
    if best_adj is None:
        # Fallback: K_{20,20} + 1 intra edge
        adj = np.zeros((n, n), dtype=bool)
        for a in A_verts:
            for b in B_verts:
                adj[a, b] = True
                adj[b, a] = True
        adj[0, 10] = True
        adj[10, 0] = True
        best_adj = adj
        best_max = _compute_max_book_fast2(best_adj, n)
    
    return best_adj, best_max

def _local_search(adj, n=40, target_edges=401, max_iters=5000, deadline=None):
    """Local search via edge swaps to reduce max book."""
    current_max = _compute_max_book_fast2(adj, n)
    best_max = current_max
    best_adj = adj.copy()
    
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        if current_max <= 6:
            break
        
        # Get all edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        
        # Get all non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    non_edges.append((i, j))
        
        if len(edges) != target_edges or len(non_edges) == 0:
            break
        
        # Pick a random edge to remove and random non-edge to add
        rem_u, rem_v = random.choice(edges)
        new_u, new_v = random.choice(non_edges)
        
        if (rem_u, rem_v) == (new_u, new_v):
            continue
        
        # Tentative swap
        adj[rem_u, rem_v] = False
        adj[rem_v, rem_u] = False
        adj[new_u, new_v] = True
        adj[new_v, new_u] = True
        
        new_max = _compute_max_book_fast2(adj, n)
        
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_adj = adj.copy()
            no_improve = 0
        else:
            # Revert
            adj[new_u, new_v] = False
            adj[new_v, new_u] = False
            adj[rem_u, rem_v] = True
            adj[rem_v, rem_u] = True
            no_improve += 1
    
    return best_adj, best_max

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0
    
    best_max = 999
    best_adj = None
    
    # Step 1: Deterministic near-bipartite construction
    adj_det, max_det = _deterministic_near_bipartite(n, target_edges)
    if max_det < best_max:
        best_max = max_det
        best_adj = adj_det.copy()
    
    # Step 2: Local search from the deterministic construction
    if time.time() < deadline - 5:
        adj_ls, max_ls = _local_search(adj_det.copy(), n, target_edges, 
                                        max_iters=3000, deadline=deadline - 5)
        if max_ls < best_max:
            best_max = max_ls
            best_adj = adj_ls.copy()
    
    # Step 3: Try circulant-based starting point with local search
    if time.time() < deadline - 5:
        adj_circ = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for k in range(1, 11):
                adj_circ[i, (i+k) % n] = True
                adj_circ[(i+k) % n, i] = True
        # Add 1 more edge
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj_circ[i, j]]
        if non_edges:
            u, v = non_edges[0]
            adj_circ[u, v] = True
            adj_circ[v, u] = True
        
        adj_circ_ls, max_circ = _local_search(adj_circ, n, target_edges,
                                               max_iters=3000, deadline=deadline - 5)
        if max_circ < best_max:
            best_max = max_circ
            best_adj = adj_circ_ls.copy()
    
    # Step 4: Additional local search from best found
    if best_adj is not None and time.time() < deadline - 3:
        adj_final, max_final = _local_search(best_adj.copy(), n, target_edges,
                                              max_iters=5000, deadline=deadline - 3)
        if max_final < best_max:
            best_max = max_final
            best_adj = adj_final.copy()
    
    # Fallback
    if best_adj is None:
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        adj[0, 10] = True
        adj[10, 0] = True
        best_adj = adj
    
    # Convert to list of [u, v]
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
