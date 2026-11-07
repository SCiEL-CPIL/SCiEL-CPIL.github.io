import numpy as np
import time

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    - Structural change: Model the problem using a Minimum Spanning Tree (MST) backbone.
    - Prioritize edges that are part of the MST or are "aligned" with the MST structure.
    - For customer-customer edges, the heuristic is inversely proportional to distance,
      multiplied by a factor that rewards edges which are short relative to their
      corresponding path length in the MST (i.e., edges that act as efficient shortcuts
      or are the tree edges themselves).
    - For depot edges, use standard inverse distance.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h

    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h

    eps = 1e-9
    coords = np.asarray(coordinates, dtype=float)
    demands_arr = np.asarray(demands, dtype=float)
    
    # Initialize heuristic matrix
    h = np.zeros((n, n))
    
    # 1. Depot Edges (i=0 or j=0)
    # Standard inverse distance
    d_depot = dm[0, :].copy()
    d_depot[0] = 1e9 # Avoid self loop
    h[0, :] = 1.0 / (d_depot + eps)
    h[:, 0] = 1.0 / (dm[:, 0] + eps)
    
    # 2. Customer-Customer Edges (i>0, j>0)
    # Extract customer subgraph
    cust_dm = dm[1:, 1:]
    n_cust = n - 1
    
    if n_cust > 0:
        # Compute MST of the customer subgraph using Prim's Algorithm (O(n^2))
        # Since n is small (50), O(n^2) is very fast.
        in_mst = [False] * n_cust
        min_edge = [np.inf] * n_cust
        parent = [-1] * n_cust
        
        # Start from node 0 in the subgraph (which is node 1 in global)
        in_mst[0] = True
        min_edge[0] = 0
        
        for _ in range(n_cust):
            # Find the min edge to a node not in MST
            u = -1
            min_val = np.inf
            for i in range(n_cust):
                if not in_mst[i] and min_edge[i] < min_val:
                    min_val = min_edge[i]
                    u = i
            
            if u == -1:
                break
                
            in_mst[u] = True
            parent_u = parent[u]
            
            # Update neighbors
            for v in range(n_cust):
                if not in_mst[v]:
                    if cust_dm[u, v] < min_edge[v]:
                        min_edge[v] = cust_dm[u, v]
                        parent[v] = u
        
        # Build adjacency list for the MST
        mst_adj = {i: [] for i in range(n_cust)}
        for i in range(n_cust):
            if parent[i] != -1:
                mst_adj[i].append(parent[i])
                mst_adj[parent[i]].append(i)
        
        # Compute pairwise distances in the MST tree
        # D_tree[i, j] = shortest path distance between i and j in the MST
        # Since it's a tree, there is a unique path. We can do BFS/DFS from each node.
        D_tree = np.zeros((n_cust, n_cust))
        for start in range(n_cust):
            # BFS/DFS to find distances from 'start'
            dist = np.full(n_cust, np.inf)
            dist[start] = 0
            visited = [False] * n_cust
            stack = [start]
            
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for neighbor in mst_adj[node]:
                    if not visited[neighbor]:
                        dist[neighbor] = dist[node] + cust_dm[node, neighbor]
                        stack.append(neighbor)
            
            D_tree[start, :] = dist
        
        # Compute heuristic for customer-customer edges
        # h[i, j] = (1 / d[i, j]) * (d[i, j] / D_tree[i, j])^p
        # If (i, j) is in MST, D_tree[i, j] == d[i, j], so ratio is 1.
        # If (i, j) is not in MST, D_tree[i, j] > d[i, j] is not necessarily true?
        # Actually, the direct edge d[i, j] is always <= the path in the MST?
        # No, the MST minimizes the sum of edges, not necessarily pairwise distances.
        # However, if d[i, j] is very small, it likely should have been in the MST.
        # If it's not in the MST, the path in the MST might be longer.
        # So d[i, j] / D_tree[i, j] <= 1 usually.
        # We want to reward edges that are "consistent" with the tree structure.
        # A ratio close to 1 means the direct edge is almost as good as the tree path.
        
        # Handle diagonal
        np.fill_diagonal(D_tree, 1.0) # Avoid division by zero, ratio will be 0 anyway if d=0? No, d[i,i]=0.
        
        # Compute ratios
        # d[i, j] is cust_dm[i, j]
        # D_tree[i, j] is the tree distance
        # We need to be careful if D_tree is inf (disconnected?), but MST is connected.
        
        # To avoid numerical issues, add eps to denominator
        safe_D_tree = D_tree + eps
        
        # Calculate alignment factor
        # If d[i,j] is much smaller than D_tree, it's a "shortcut".
        # If d[i,j] is close to D_tree, it's "aligned".
        # The directive says: "prioritizing edges that are part of or close to the global connectivity backbone"
        # Edges in the backbone have ratio 1.
        # Edges that are shortcuts have ratio < 1.
        # We might want to penalize shortcuts slightly if they break the "backbone" flow?
        # Or maybe shortcuts are good?
        # In TSP, the optimal tour is a Hamiltonian cycle. The MST is a subgraph.
        # Edges in the MST are "safe" choices.
        # Let's use: h = (1/d) * (ratio)^2.
        # This heavily favors MST edges (ratio=1) over non-MST edges (ratio<1).
        
        ratios = cust_dm / safe_D_tree
        # Clip ratios to [0, 1] to handle any floating point noise where ratio > 1
        ratios = np.clip(ratios, 0.0, 1.0)
        
        # Inverse distance
        inv_d = 1.0 / (cust_dm + eps)
        
        # Combined heuristic
        h_cust = inv_d * (ratios ** 2)
        
        # Assign to main matrix
        h[1:, 1:] = h_cust

    # Ensure diagonal is 0
    np.fill_diagonal(h, 0.0)
    
    # Handle non-finite or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Normalize to prevent extreme values
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
        
    # Re-apply positivity constraint
    h[h <= 0] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    return h
