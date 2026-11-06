import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n = 40
    target_edges = 401
    rng = np.random.RandomState(42)
    
    def compute_books(adj_int):
        """Compute book sizes for all edges using matrix multiply.
        adj_int is integer adjacency matrix (0/1), symmetric, zero diagonal.
        Returns array where books[i,j] = number of common neighbors of i,j
        (i.e., the book size for edge (i,j) if adj[i,j]=1)."""
        A2 = adj_int @ adj_int
        # A2[i,j] = sum_k A[i,k]*A[k,j] = number of common neighbors
        # This includes the edge itself if it exists? No: A[i,k]*A[k,j] for k=i gives A[i,i]*A[i,j]=0
        # For k=j: A[i,j]*A[j,j]=0. So A2[i,j] is exactly the number of common neighbors.
        return A2
    
    def score_graph(adj_int):
        """Return max book size over all edges present in adj_int."""
        A2 = compute_books(adj_int)
        # Only consider edges that exist
        mask = adj_int > 0
        if not mask.any():
            return 0
        np.fill_diagonal(A2, 0)
        # Set non-edge positions to -1 so they don't affect max
        A2_masked = np.where(mask, A2, -1)
        return int(A2_masked.max())
    
    def random_initial_graph():
        """Generate a random graph with exactly target_edges edges."""
        adj = np.zeros((n, n), dtype=np.int32)
        # Start with a random subset of edges
        all_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                all_pairs.append((i, j))
        # Shuffle and pick target_edges
        idx = rng.permutation(len(all_pairs))[:target_edges]
        for k in idx:
            i, j = all_pairs[k]
            adj[i][j] = 1
            adj[j][i] = 1
        return adj
    
    def generate_balanced_initial():
        """Generate a near-bipartite graph with some internal edges spread out."""
        adj = np.zeros((n, n), dtype=np.int32)
        # Split into two groups of 20
        # Add all cross edges except remove a few to make room
        cross_count = 0
        for a in range(20):
            for b in range(20, 40):
                adj[a][b] = 1
                adj[b][a] = 1
                cross_count += 1
        # cross_count = 400, need 1 more edge total = 401
        # Add one internal edge
        adj[0][1] = 1
        adj[1][0] = 1
        return adj
    
    def try_swap(adj, remove_edge, add_edge):
        """Try removing remove_edge and adding add_edge. Return new adj and score delta."""
        i, j = remove_edge
        k, l = add_edge
        if adj[i][j] == 0:
            return None, None
        if adj[k][l] == 1:
            return None, None
        # Make the swap
        adj[i][j] = 0
        adj[j][i] = 0
        adj[k][l] = 1
        adj[l][k] = 1
        return adj, None
    
    def revert_swap(adj, remove_edge, add_edge):
        i, j = remove_edge
        k, l = add_edge
        adj[i][j] = 1
        adj[j][i] = 1
        adj[k][l] = 0
        adj[l][k] = 0
    
    # Phase 1: Generate multiple initial candidates and pick best
    candidates = []
    
    # Candidate 1: Balanced bipartite + 1
    adj1 = generate_balanced_initial()
    candidates.append(adj1)
    
    # Candidate 2: Random graph
    adj2 = random_initial_graph()
    candidates.append(adj2)
    
    # Candidate 3: Near-bipartite with some internal edges spread
    adj3 = np.zeros((n, n), dtype=np.int32)
    # Add 390 cross edges (remove 10 random cross edges)
    cross_edges = [(a, b) for a in range(20) for b in range(20, 40)]
    rng.shuffle(cross_edges)
    for i, j in cross_edges[:390]:
        adj3[i][j] = 1
        adj3[j][i] = 1
    # Add 11 internal edges spread across both parts
    internal_a = [(a, b) for a in range(20) for b in range(a+1, 20)]
    internal_b = [(a, b) for a in range(20, 40) for b in range(a+1, 40)]
    rng.shuffle(internal_a)
    rng.shuffle(internal_b)
    for i, j in internal_a[:5]:
        adj3[i][j] = 1
        adj3[j][i] = 1
    for i, j in internal_b[:6]:
        adj3[i][j] = 1
        adj3[j][i] = 1
    candidates.append(adj3)
    
    # Evaluate all candidates
    best_adj = None
    best_score = float('inf')
    for adj_c in candidates:
        s = score_graph(adj_c)
        if s < best_score:
            best_score = s
            best_adj = adj_c.copy()
    
    # Phase 2: Local search with vectorized scoring
    adj = best_adj.copy()
    current_score = best_score
    
    # Precompute edge list for current graph
    def get_edges(adj_int):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_int[i][j]:
                    edges.append((i, j))
        return edges
    
    def get_non_edges(adj_int):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj_int[i][j]:
                    non_edges.append((i, j))
        return non_edges
    
    edges_list = get_edges(adj)
    non_edges_list = get_non_edges(adj)
    
    # Local search: try random swaps
    max_iterations = 50000
    iter_count = 0
    no_improve_count = 0
    patience = 2000
    
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        
        iter_count += 1
        
        # Pick a random edge to remove and a random non-edge to add
        if not edges_list or not non_edges_list:
            break
        
        e_idx = rng.randint(len(edges_list))
        ne_idx = rng.randint(len(non_edges_list))
        
        rem_edge = edges_list[e_idx]
        add_edge = non_edges_list[ne_idx]
        
        # Perform swap
        ri, rj = rem_edge
        ai, aj = add_edge
        
        adj[ri][rj] = 0
        adj[rj][ri] = 0
        adj[ai][aj] = 1
        adj[aj][ai] = 1
        
        new_score = score_graph(adj)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
            # Update edge lists
            edges_list[e_idx] = add_edge
            non_edges_list[ne_idx] = rem_edge
            no_improve_count = 0
        else:
            # Revert
            adj[ri][rj] = 1
            adj[rj][ri] = 1
            adj[ai][aj] = 0
            adj[aj][ai] = 0
            no_improve_count += 1
            if no_improve_count > patience:
                # Reset patience and try a different strategy
                no_improve_count = 0
                # Occasionally do a 2-swap: remove 2 edges, add 2 edges
                if len(edges_list) >= 2 and len(non_edges_list) >= 2:
                    e1_idx = rng.randint(len(edges_list))
                    e2_idx = rng.randint(len(edges_list))
                    ne1_idx = rng.randint(len(non_edges_list))
                    ne2_idx = rng.randint(len(non_edges_list))
                    
                    # Ensure all distinct
                    if len(set([e1_idx, e2_idx, ne1_idx, ne2_idx])) < 4:
                        continue
                    
                    e1 = edges_list[e1_idx]
                    e2 = edges_list[e2_idx]
                    ne1 = non_edges_list[ne1_idx]
                    ne2 = non_edges_list[ne2_idx]
                    
                    # Check no conflicts
                    if e1 == e2 or ne1 == ne2:
                        continue
                    
                    # Perform 2-swap
                    for (u, v) in [e1, e2]:
                        adj[u][v] = 0
                        adj[v][u] = 0
                    for (u, v) in [ne1, ne2]:
                        adj[u][v] = 1
                        adj[v][u] = 1
                    
                    new_score = score_graph(adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = adj.copy()
                        # Update lists
                        edges_list[e1_idx] = ne1
                        edges_list[e2_idx] = ne2
                        non_edges_list[ne1_idx] = e1
                        non_edges_list[ne2_idx] = e2
                        no_improve_count = 0
                    else:
                        # Revert
                        for (u, v) in [e1, e2]:
                            adj[u][v] = 1
                            adj[v][u] = 1
                        for (u, v) in [ne1, ne2]:
                            adj[u][v] = 0
                            adj[v][u] = 0
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    # Verify we have exactly 401 edges
    if len(edges) != target_edges:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return edges
