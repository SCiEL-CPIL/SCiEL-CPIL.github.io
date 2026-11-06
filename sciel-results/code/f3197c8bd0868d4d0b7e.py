"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    n = 40
    
    def time_ok():
        return time.time() < deadline
    
    def time_left():
        return deadline - time.time()
    
    # Bitset-based adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def can_add(adj, u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        Optimized: iterate over the smaller neighbor set in the outer loop.
        C6: u-v-x-y-z-w-u, all distinct.
        """
        Nv = adj[v]
        Nu = adj[u]
        
        # To minimize work, iterate x over the smaller set
        if Nv.bit_count() <= Nu.bit_count():
            # x in N(v), w in N(u)
            x_mask = Nv
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                if x == u:
                    continue
                w_mask = Nu
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    if w == v or w == x:
                        continue
                    # Need y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        else:
            # deg(u) < deg(v): x in N(u), w in N(v)
            # C6: u-v-w-z-y-x-u (renamed)
            # x in N(u), w in N(v), y in N(x), z in N(w), y~z
            x_mask = Nu
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                if x == v:
                    continue
                w_mask = Nv
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    if w == u or w == x:
                        continue
                    # y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        return True
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges
    
    # Spectral-guided initialization
    def spectral_init():
        """Create a random 4-regular graph, compute eigenvector, use it to guide edge selection."""
        rng = np.random.default_rng(42)
        
        # Create a random 4-regular graph on 40 vertices
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
            deg = adj_matrix[i].sum()
            if deg < 4:
                for j in range(n):
                    if i != j and adj_matrix[i][j] == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        deg += 1
                        if deg >= 4:
                            break
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        
        # Find eigenvalue closest to sqrt(5) ≈ 2.236
        target = np.sqrt(5)
        idx = np.argmin(np.abs(eigenvalues - target))
        spec_vec = eigenvectors[:, idx]
        
        # Use spectral components to guide edge inclusion
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                weights[i][j] = abs(spec_vec[i] * spec_vec[j])
        
        # Get all potential edges sorted by weight (descending)
        potential_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                potential_edges.append((weights[i][j], i, j))
        potential_edges.sort(reverse=True)
        
        return potential_edges, adj_matrix
    
    # Greedy addition using fast C6 check
    def greedy_add(adj, edge_set, edges, candidate_edges):
        """Add all non-conflicting edges from candidate_edges."""
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            for item in candidate_edges:
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added_any = True
                else:
                    remaining.append(item)
            candidate_edges = remaining
        return edges
    
    # Local search with edge removal to escape local maxima
    def local_search(adj, edge_set, edges, max_iters=200):
        """Local search: try to add edges, remove edges to escape local maxima."""
        no_improve_count = 0
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for _ in range(max_iters):
            if time_left() < 5:
                break
            
            added = False
            
            # Try to add a random non-edge
            for _ in range(30):
                if time_left() < 5:
                    break
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    continue
                
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added = True
                    no_improve_count = 0
                    break
            
            if not added:
                # Escape local maximum by removing an edge
                if no_improve_count > 10:
                    # Find a random edge to remove
                    edge_list = [(u, v) for u, v in edge_set]
                    if edge_list:
                        u, v = random.choice(edge_list)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove((u, v))
                        edges = [e for e in edges if not (min(e[0],e[1]) == min(u,v) and max(e[0],e[1]) == max(u,v))]
                        no_improve_count = 0
                    else:
                        break
                
                no_improve_count += 1
        
        return edges
    
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Spectral-guided initialization
    potential_edges, _ = spectral_init()
    
    edges1 = []
    adj1 = [0] * n
    edge_set1 = set()
    greedy_add(adj1, edge_set1, edges1, potential_edges)
    
    if len(edges1) > best_count:
        best_count = len(edges1)
        best_edges = edges1[:]
    
    # Strategy 2: Standard K5-based initialization
    edges2 = []
    for k in range(8):
        edges2 += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    adj2 = make_adj(edges2)
    edge_set2 = set((min(u,v), max(u,v)) for u, v in edges2)
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set2]
    greedy_add(adj2, edge_set2, edges2, all_possible)
    
    if len(edges2) > best_count:
        best_count = len(edges2)
        best_edges = edges2[:]
    
    # Strategy 3: Randomized restarts from best
    for restart in range(100):
        if not time_ok():
            break
        
        if restart < 5:
            # Start from best
            current_edges = best_edges[:]
        else:
            # Start from a random subset of best
            current_edges = [e for e in best_edges if random.random() < 0.85]
        
        current_edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)
        
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in current_edge_set]
        random.shuffle(remaining)
        
        greedy_add(current_adj, current_edge_set, current_edges, remaining)
        
        # Local search to improve
        current_edges = local_search(current_adj, current_edge_set, current_edges, max_iters=50)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    # Final polish: try to add any remaining edges
    final_edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
    final_adj = make_adj(best_edges)
    final_remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in final_edge_set]
    greedy_add(final_adj, final_edge_set, best_edges, final_remaining)
    
    # Final local search
    best_edges = local_search(final_adj, final_edge_set, best_edges, max_iters=100)
    
    return best_edges
# EVOLVE-BLOCK-END
