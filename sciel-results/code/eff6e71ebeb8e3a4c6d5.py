import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    random.seed(42)
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_edges = len(all_edges)
    
    # Build edge index for quick lookup
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def get_adjacency(adj_matrix):
        """Return adjacency matrix as numpy array."""
        return adj_matrix
    
    def is_k33_free(adj_matrix):
        """Check if graph is K3,3-free using vectorized operations."""
        # For each triple of vertices, check if they have 3+ common neighbors
        # K3,3 exists iff there exist two disjoint sets A,B of size 3
        # such that every vertex in A is connected to every vertex in B
        # Equivalent: there exist 6 distinct vertices a1,a2,a3,b1,b2,b3
        # where ai-bj edges all exist for all i,j
        
        # More efficient: A K3,3 subgraph exists iff there are two vertices
        # with at least 3 common neighbors that also form a K3,3 structure.
        # Actually, the cleanest check: for each pair of vertices (u,v),
        # count common neighbors. If any pair has >=3 common neighbors AND
        # those common neighbors contain a triangle with connections... 
        # 
        # Wait - the problem says "no three vertices have three common neighbours".
        # This means: for any three vertices {x,y,z}, |N(x) ∩ N(y) ∩ N(z)| < 3
        # This is actually a stronger condition than just K3,3-free in general,
        # but let me re-read: "no K3,3" and "no three vertices have three common neighbours"
        # 
        # A K3,3 has bipartition (a,b,c) and (d,e,f). The three vertices d,e,f 
        # have common neighbors a,b,c (at least). So if no three vertices share 
        # 3+ common neighbors, there's no K3,3. But the converse isn't quite right either.
        # 
        # Actually for K3,3: pick any 3 vertices from one part, they all share 
        # the other 3 as common neighbors. So checking that no 3 vertices have 
        # 3+ common neighbors is sufficient to ensure K3,3-free.
        
        # Vectorized check: For each triple, compute intersection of neighborhoods
        # This is O(n^3 * n) = O(n^4) which for n=20 is 160000 operations, fine.
        
        adj = adj_matrix.astype(np.int8)
        
        # For efficiency, check pairs first
        # If two vertices u,v have >=3 common neighbors, we need to check if
        # any 3 of those common neighbors share another common neighbor with u and v
        # 
        # Simpler: iterate over all triples and check common neighbor count
        # n=20, so C(20,3)=1140 triples, each requiring intersection of 3 bitsets
        # Using numpy: 1140 * 20 operations is trivial
        
        for i in range(n):
            for j in range(i + 1, n):
                # Common neighbors of i and j
                common_ij = adj[i] & adj[j]
                common_count_ij = np.sum(common_ij)
                if common_count_ij < 3:
                    continue
                # Now for each k not in {i,j} and not a common neighbor of i,j
                # Check if k also shares >=3 common neighbors with i and j
                # Actually we need: |N(i) ∩ N(j) ∩ N(k)| >= 3
                for k in range(j + 1, n):
                    if k == i or k == j:
                        continue
                    common_ijk = common_ij & adj[k]
                    if np.sum(common_ijk) >= 3:
                        return False
        return True
    
    # Start with a random dense graph and remove edges to satisfy constraint
    # Strategy: Start with all edges, then greedily remove violating edges
    
    def remove_violating_edges(adj_matrix):
        """Remove edges until K3,3-free. Returns (adj_matrix, num_removed)."""
        removed = 0
        changed = True
        while changed:
            changed = False
            # Find a violating triple
            for i in range(n):
                for j in range(i + 1, n):
                    common_ij = adj_matrix[i] & adj_matrix[j]
                    if np.sum(common_ij) < 3:
                        continue
                    for k in range(j + 1, n):
                        common_ijk = common_ij & adj_matrix[k]
                        if np.sum(common_ijk) >= 3:
                            # Found violation: vertices i,j,k share >=3 common neighbors
                            # Remove an edge incident to one of i,j,k to break the violation
                            # Strategy: remove an edge from the common neighbors
                            common_list = np.where(common_ijk)[0]
                            # Remove edge between i and a common neighbor
                            cn = common_list[0]
                            if adj_matrix[i, cn]:
                                adj_matrix[i, cn] = 0
                                adj_matrix[cn, i] = 0
                                removed += 1
                                changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj_matrix, removed
    
    # Approach 1: Start with complete graph, remove edges
    best_edge_count = -1
    best_adj = None
    
    # Try multiple random initializations
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Start with a random subset of edges (dense bias)
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Add random edges with probability 0.8 initially
        for edge in all_edges:
            if random.random() < 0.8:
                u, v = edge
                adj[u, v] = 1
                adj[v, u] = 1
        
        # Remove violating edges
        adj, removed = remove_violating_edges(adj)
        edge_count = np.sum(adj) // 2
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj.copy()
        
        # Local search: try to add back edges
        if time.time() > deadline - 10.0:
            break
        
        # Try adding edges that were removed or not present
        current_adj = adj.copy()
        current_count = edge_count
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try all possible edges not currently present
            for edge in random.sample(all_edges, min(len(all_edges), 500)):
                u, v = edge
                if current_adj[u, v]:
                    continue
                # Try adding this edge
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                if is_k33_free(current_adj):
                    current_count += 1
                    improved = True
                    if current_count > best_edge_count:
                        best_edge_count = current_count
                        best_adj = current_adj.copy()
                else:
                    # Revert
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
        
        # Edge removal + re-addition local search
        if time.time() > deadline - 10.0:
            break
            
        current_adj = best_adj.copy()
        current_count = best_edge_count
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            # Remove a random edge
            present_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i, j]]
            if not present_edges:
                break
            u, v = random.choice(present_edges)
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_count -= 1
            
            # Try to add back different edges
            for edge in random.sample(all_edges, min(len(all_edges), 200)):
                a, b = edge
                if current_adj[a, b]:
                    continue
                current_adj[a, b] = 1
                current_adj[b, a] = 1
                if is_k33_free(current_adj):
                    current_count += 1
                    if current_count > best_edge_count:
                        best_edge_count = current_count
                        best_adj = current_adj.copy()
                    break
                else:
                    current_adj[a, b] = 0
                    current_adj[b, a] = 0
    
    # Build final edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges
