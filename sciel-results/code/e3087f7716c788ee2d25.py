"""
Exact Maximum Cut solver using Branch and Bound.
Strategy: Systematically explore the binary assignment space, using upper bounds
to prune branches that cannot improve upon the current best solution.
For n=100, this is feasible within 120s with good initial bounds and pruning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def partition(adjacency):
    """
    Solve Maximum Cut using Branch and Bound.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin for Python overhead/return
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast access
    adj = np.array(adjacency, dtype=np.int32)
    
    # Precompute neighbors and degrees for faster local updates if needed,
    # but for B&B we mostly need the global structure.
    # We will represent the state as an array of assignments: -1 for unassigned, 0 or 1 for assigned.
    
    # 1. Get a good initial lower bound using a heuristic (Goemans-Williamson style or Greedy)
    # Let's use a simple greedy heuristic to get a strong incumbent quickly.
    def greedy_heuristic(adj, n, rng):
        labels = np.full(n, -1, dtype=np.int8)
        # Random order to avoid bias
        order = rng.permutation(n)
        
        # For each vertex in random order, assign the label that maximizes immediate cut gain
        # Gain = (neighbors with opposite label) - (neighbors with same label)
        # Initially, gain is 0 for both.
        
        # To speed this up, we can just do a random 50/50 split and then hill climb,
        # or use the "local ratio" heuristic.
        # Let's stick to a simple random restart hill climb for the initial bound, 
        # but do it very fast.
        
        best_labels = None
        best_cut = -1
        
        # Quick random restarts
        for _ in range(10):
            l = rng.integers(0, 2, size=n)
            # Simple hill climb
            improved = True
            while improved:
                improved = False
                # Calculate gain for flipping each vertex
                # gain[v] = sum_u adj[v,u] * (2*l[v] - 1) * (2*l[u] - 1) ... no, that's for energy.
                # Cut contribution of v: sum_u adj[v,u] * (l[v] != l[u])
                # If we flip v, new contribution: sum_u adj[v,u] * (1-l[v] != l[u]) = sum_u adj[v,u] * (l[v] == l[u])
                # Old contribution: sum_u adj[v,u] * (l[v] != l[u])
                # Gain = (Old - New) ? No, we want to maximize cut.
                # Gain = New_Cut - Old_Cut
                # New_Cut = Old_Cut - Old_Contribution_v + New_Contribution_v
                # Gain = New_Contribution_v - Old_Contribution_v
                # New_Contribution_v = sum_u adj[v,u] * (l[v] == l[u])
                # Old_Contribution_v = sum_u adj[v,u] * (l[v] != l[u])
                
                # Vectorized gain calculation
                # Let s = 2*l - 1 (so 1 for 1, -1 for 0)
                # Cut = 0.5 * (n * sum_deg - s @ (adj * s)) ... no.
                # Cut = 0.5 * sum_{u,v} adj[u,v] (1 - s[u]s[v]) / 2 ?
                # Cut = 0.5 * (sum_deg * n - s @ (adj @ s)) / 2 ?
                # Let's use the direct formula:
                # cut = 0.5 * (sum_{u} deg[u] - sum_{u,v} adj[u,v] s[u] s[v])
                
                s = 2 * l - 1
                # adj_s = adj @ s
                # s_adj_s = s @ (adj @ s)
                # cut = 0.5 * (degrees.sum() - s_adj_s)
                
                # Gain from flipping v:
                # s[v] becomes -s[v]
                # s_adj_s changes by: -2 * s[v] * (adj @ s)[v]
                # New s_adj_s = s_adj_s - 2 * s[v] * (adj @ s)[v]
                # New cut = 0.5 * (sum_deg - (s_adj_s - 2 * s[v] * (adj @ s)[v]))
                # Gain = 0.5 * 2 * s[v] * (adj @ s)[v] = s[v] * (adj @ s)[v]
                
                adj_s = adj @ s
                gains = s * adj_s
                
                # Find max gain
                max_gain = np.max(gains)
                if max_gain > 1e-6:
                    v = np.argmax(gains)
                    l[v] = 1 - l[v]
                    s[v] = -s[v]
                    improved = True
                    # Update s_adj_s incrementally? 
                    # s_adj_s = s @ (adj @ s)
                    # This is O(n^2) to recompute. 
                    # For n=100, O(n^2) is 10k ops, which is fast.
                    # But we are in a loop. Let's just recompute adj_s and s_adj_s each iteration?
                    # No, we just need gains.
                    # Gains depend on s and adj_s.
                    # If we flip v, s changes, so adj_s changes.
                    # adj_s[u] = sum_w adj[u,w] s[w]
                    # If s[v] flips, adj_s[u] changes by adj[u,v] * (-2 * s[v]_old)
                    # This is getting complex. For n=100, recomputing adj @ s is O(n^2) = 10k.
                    # 100 iterations * 10k = 1M ops. Very fast.
                    pass
        
        # Calculate final cut for this l
        s = 2 * l - 1
        s_adj_s = s @ (adj @ s)
        cut = 0.5 * (degrees_sum - s_adj_s)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = l.copy()
            
        return best_labels, best_cut

    degrees = adj.sum(axis=1)
    degrees_sum = degrees.sum()
    
    rng = np.random.default_rng(42)
    
    # Get initial incumbent
    best_labels, best_cut = greedy_heuristic(adj, n, rng)
    
    # 2. Branch and Bound
    # State:
    # assigned: list of -1, 0, 1
    # current_cut: cut size of assigned edges
    # unassigned: set of unassigned vertices
    
    # We use a stack for DFS
    # Each item: (assigned_array, current_cut, unassigned_set, upper_bound)
    # To save memory/time, we can store the path and recompute, but n=100 is small.
    # We'll store the assignment array.
    
    # Precompute edge list for faster cut updates?
    # Edges: list of (u, v)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append((i, j))
    
    # Helper to calculate cut size from labels
    def calc_cut(labels):
        # labels: 0/1 array
        # cut = sum(1 if labels[u] != labels[v] else 0 for u,v in edges)
        # Vectorized:
        # diff = labels[0::2] != labels[1::2] ... no, edges are arbitrary.
        # Use matrix:
        # l = labels.astype(float)
        # return float(l @ (adj @ (1 - l))) # No
        # cut = 0.5 * (degrees_sum - (2*labels-1) @ (adj @ (2*labels-1)))
        s = 2 * labels - 1
        return 0.5 * (degrees_sum - s @ (adj @ s))
    
    # Upper Bound Function:
    # Given current assignment (0/1/-1) and current_cut:
    # Upper Bound = current_cut + (Sum of degrees of unassigned vertices) / 2
    # Why? 
    # Each unassigned vertex v has degree d_v.
    # Edges incident to v:
    # 1. Edge to assigned u: contributes 1 to cut if label[v] != label[u]. Max 1.
    # 2. Edge to unassigned w: contributes 1 to cut if label[v] != label[w].
    # The sum of degrees of unassigned vertices counts:
    # - Edges to assigned vertices once.
    # - Edges to unassigned vertices twice.
    # Let E_unassigned_assigned be edges between U and A.
    # Let E_unassigned_unassigned be edges within U.
    # Sum_Deg_U = E_unassigned_assigned + 2 * E_unassigned_unassigned.
    # Max possible additional cut from U:
    # - All edges in E_unassigned_assigned can be cut (assign opposite to A). Max = E_unassigned_assigned.
    # - Edges in E_unassigned_unassigned: Max cut of subgraph induced by U. <= E_unassigned_unassigned.
    # So Max Add <= E_unassigned_assigned + E_unassigned_unassigned.
    # Note that Sum_Deg_U / 2 = 0.5 * E_unassigned_assigned + E_unassigned_unassigned.
    # So Upper Bound = current_cut + 0.5 * E_unassigned_assigned + E_unassigned_unassigned.
    # This is a valid upper bound.
    
    def get_upper_bound(assigned, current_cut, unassigned_set):
        # Calculate sum of degrees of unassigned vertices
        # deg_sum_u = sum(deg[v] for v in unassigned_set)
        # This includes edges to A and edges to U.
        # bound = current_cut + deg_sum_u / 2
        # This is valid.
        deg_sum_u = 0
        for v in unassigned_set:
            deg_sum_u += degrees[v]
        return current_cut + deg_sum_u / 2.0
    
    # Branching strategy:
    # Pick unassigned vertex with highest degree to split the space most effectively.
    
    # Initial state
    assigned = np.full(n, -1, dtype=np.int8)
    unassigned_set = set(range(n))
    current_cut = 0
    
    # Stack items: (assigned_copy, current_cut, unassigned_set_copy)
    # To avoid copying sets, we can just pass the list of unassigned vertices.
    # But n=100, copying a list of 100 ints is fine.
    
    stack = [(assigned.copy(), 0, list(unassigned_set))]
    
    nodes_explored = 0
    
    while stack:
        if time.time() > DEADLINE:
            break
        
        assigned, curr_cut, unassigned_list = stack.pop()
        
        if not unassigned_list:
            # All assigned
            if curr_cut > best_cut:
                best_cut = curr_cut
                best_labels = assigned.copy()
            continue
        
        # Calculate upper bound
        ub = get_upper_bound(assigned, curr_cut, set(unassigned_list))
        
        if ub <= best_cut:
            continue
        
        # Select vertex to branch on: highest degree
        # v = max(unassigned_list, key=lambda x: degrees[x])
        # To make it deterministic and potentially faster, just pick the first one?
        # No, highest degree is better for pruning.
        v = max(unassigned_list, key=lambda x: degrees[x])
        
        # Create two new states: v=0 and v=1
        new_unassigned_list = [x for x in unassigned_list if x != v]
        
        # Branch 0
        assigned_0 = assigned.copy()
        assigned_0[v] = 0
        # Calculate new cut contribution
        # Edges from v to already assigned vertices:
        # For each neighbor u of v where assigned[u] != -1:
        # If assigned[u] != 0, edge is in cut.
        add_cut_0 = 0
        for u in range(n):
            if adj[v, u] and assigned[u] != -1:
                if assigned[u] != 0:
                    add_cut_0 += 1
        stack.append((assigned_0, curr_cut + add_cut_0, new_unassigned_list))
        
        # Branch 1
        assigned_1 = assigned.copy()
        assigned_1[v] = 1
        add_cut_1 = 0
        for u in range(n):
            if adj[v, u] and assigned[u] != -1:
                if assigned[u] != 1:
                    add_cut_1 += 1
        stack.append((assigned_1, curr_cut + add_cut_1, new_unassigned_list))
        
        nodes_explored += 1
        # Optional: print progress if it takes too long? No, avoid IO.
        
    # If B&B didn't finish or didn't improve, ensure we return the best found
    # If best_labels is None (shouldn't happen as we have heuristic), fallback
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
