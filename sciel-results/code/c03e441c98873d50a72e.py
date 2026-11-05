import numpy as np
import time
import sys

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return {}
    if n == 1:
        return {0: [0]} # Dummy
    if n == 2:
        return {0: [1], 1: [0]}
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    # adj list for the tree structure
    adj = {i: [] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
        r = d_active.sum(axis=1)
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        new_id = next_id
        next_id += 1
        
        # Update adjacency
        # best_i and best_j are connected to new_id
        # Remove edge between best_i and best_j? 
        # NJ doesn't keep a single edge between them, they are joined via new node.
        # In the current active set, best_i and best_j are just nodes.
        # We need to update the "virtual" tree.
        
        # Let's just build the final tree structure at the end or maintain it.
        # Maintaining it is better for SPR later.
        
        # In NJ, we create a new node connected to best_i and best_j.
        # We need to know who was connected to best_i and best_j previously?
        # No, NJ merges two clusters.
        
        # Let's stick to the standard NJ tree construction logic but record the tree.
        
        # Actually, for SPR, we need the full tree.
        # Let's use a standard library or a robust implementation.
        # Since I can't import external phylo libs, I'll implement a simple tree builder.
        
        # For now, let's just return the Newick string from NJ and parse it for SPR?
        # No, parsing is slow.
        
        # Let's modify _neighbor_joining to return an adjacency list.
        
        # Current adj logic:
        # When we merge i and j into new_node:
        # 1. Find all neighbors of i and j in the current tree?
        # This is tricky because NJ is a "star" of distances, not a graph initially.
        
        # Alternative: Use the `trees` dict approach to build Newick, then parse.
        # Given n<=32, parsing a Newick string is fast enough.
        
        pass

def _parse_newick_to_adj(newick, n_taxa):
    """Parse Newick to adjacency list. Returns adj dict, leaf_map (id->label), internal_count."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_map = {}
    next_id = 0
    is_leaf = {}
    
    def parse(node_str):
        nonlocal next_id
        node_str = node_str.strip()
        if not node_str:
            return None
        
        if node_str.startswith('('):
            my_id = next_id
            next_id += 1
            adj[my_id] = []
            is_leaf[my_id] = False
            
            # Find matching close paren
            depth = 0
            i = 0
            while i < len(node_str):
                if node_str[i] == '(':
                    depth += 1
                elif node_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = node_str[1:i]
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            for c in children:
                child_id = parse(c)
                if child_id is not None:
                    adj[my_id].append(child_id)
                    adj[child_id].append(my_id)
            
            return my_id
        else:
            label = node_str.split(':')[0].strip()
            my_id = next_id
            next_id += 1
            adj[my_id] = []
            is_leaf[my_id] = True
            leaf_map[my_id] = label
            return my_id
    
    root = parse(newick)
    return adj, leaf_map, is_leaf

def _get_leaf_distances(adj, leaf_map, n_leaves):
    """Compute all-pairs leaf distances in the tree (unweighted, topological only)."""
    leaf_ids = list(leaf_map.keys())
    leaf_idx = {l: i for i, l in enumerate(leaf_ids)}
    dist = np.zeros((n_leaves, n_leaves))
    
    # BFS from each leaf
    for start_idx, start_node in enumerate(leaf_ids):
        visited = {start_node: 0}
        stack = [start_node]
        while stack:
            node = stack.pop()
            d = visited[node]
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    stack.append(neighbor)
        
        for end_idx, end_node in enumerate(leaf_ids):
            dist[start_idx, end_idx] = visited[end_node]
            
    return dist

def _calculate_ssr(adj, leaf_map, dist_obs):
    """Calculate Sum of Squared Residuals for the tree topology."""
    n_leaves = len(leaf_map)
    if n_leaves < 3:
        return 0.0
    
    # Get topological distances
    dist_topo = _get_leaf_distances(adj, leaf_map, n_leaves)
    
    # Build A matrix
    # Rows: pairs of leaves (i, j) where i < j
    # Cols: internal edges
    # A[p, e] = 1 if edge e is on path between leaf i and j
    
    # Identify internal edges
    # An edge is internal if it connects two internal nodes?
    # No, in an unrooted tree, any edge can be a "branch" in the linear model.
    # The linear model is: d_ij = sum_{e in path(i,j)} b_e
    # We need to solve for b_e for all edges e.
    # There are n-3 independent edges (dimension of the space).
    # We can pick any n-3 edges as the basis, or just use all edges and solve the least squares problem with a rank-deficient A?
    # Better: Use the n-3 internal edges (edges not incident to a leaf? No, that's not right for unrooted).
    # Actually, the dimension of the space of trees is n-3.
    # We can select a set of n-3 edges that form a basis.
    # A simple basis: remove one leaf's incident edge? No.
    # Standard approach: The matrix A has rank n-3.
    # We can use SVD or pinv, or just pick n-3 columns.
    
    # Let's identify all edges.
    edges = []
    seen = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    # For each edge, determine which leaf pairs it connects.
    # This is O(E * N^2) which is fine.
    
    # A matrix: n_pairs x n_edges
    n_pairs = n_leaves * (n_leaves - 1) // 2
    A = np.zeros((n_pairs, len(edges)))
    
    leaf_ids = list(leaf_map.keys())
    leaf_idx = {l: i for i, l in enumerate(leaf_ids)}
    
    # For each edge, find the split it induces
    # Remove edge (u, v), find component of v.
    # If leaf i is in component and leaf j is not, then edge is on path.
    
    edge_splits = []
    for u, v in edges:
        # Find component of v without u
        comp = set()
        stack = [v]
        visited = {v}
        while stack:
            node = stack.pop()
            comp.add(node)
            for neighbor in adj[node]:
                if neighbor != u and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        # Check which leaves are in comp
        leaves_in_comp = [leaf_map[node] for node in comp if node in leaf_map]
        split_set = set(leaves_in_comp)
        edge_splits.append(split_set)
    
    # Fill A
    pair_idx = 0
    for i in range(n_leaves):
        for j in range(i+1, n_leaves):
            li = leaf_ids[i]
            lj = leaf_ids[j]
            for e_idx, split in enumerate(edge_splits):
                # Edge is on path if exactly one of li, lj is in split
                if (li in split) != (lj in split):
                    A[pair_idx, e_idx] = 1.0
            pair_idx += 1
    
    # Observed distances vector
    d_obs_vec = []
    for i in range(n_leaves):
        for j in range(i+1, n_leaves):
            d_obs_vec.append(dist_obs[i, j])
    d_obs_vec = np.array(d_obs_vec)
    
    # Solve A b = d_obs
    # Use least squares
    try:
        b, residuals, rank, s = np.linalg.lstsq(A, d_obs_vec, rcond=None)
        # Calculate SSR
        pred = A @ b
        ssr = np.sum((pred - d_obs_vec) ** 2)
        return ssr
    except:
        return float('inf')

def _spr_refine(adj, leaf_map, dist_obs, max_passes=10):
    """Perform SPR refinement to minimize SSR."""
    current_ssr = _calculate_ssr(adj, leaf_map, dist_obs)
    n_leaves = len(leaf_map)
    if n_leaves < 3:
        return adj, current_ssr
    
    # Identify internal edges (edges where both endpoints are internal? Or just all edges?)
    # In unrooted tree, all edges are valid for pruning.
    # But pruning a leaf edge is trivial (just moves the leaf).
    # We should focus on internal edges.
    
    # Let's just iterate all edges.
    
    for _ in range(max_passes):
        improved = False
        # Get list of edges
        edges = []
        seen = set()
        for u in adj:
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        for i, (u, v) in enumerate(edges):
            # Prune v from u
            # Find component of v
            comp_v = set()
            stack = [v]
            visited = {v}
            while stack:
                node = stack.pop()
                comp_v.add(node)
                for neighbor in adj[node]:
                    if neighbor != u and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            # Try regrafting onto all other edges
            for j, (a, b) in enumerate(edges):
                if i == j:
                    continue
                # Check if (a, b) is inside comp_v
                if a in comp_v and b in comp_v:
                    continue
                # If one is in comp_v and the other is not, it's the edge (u, v) itself?
                # We already skipped i==j.
                # If a in comp_v, then b must be outside (since it's a different edge).
                # This is a valid regrafting spot?
                # Actually, regrafting onto an edge that is "adjacent" to the pruned part is effectively a 1-1 swap or similar.
                # Let's just try all valid external edges.
                
                # Construct new tree
                new_adj = {k: list(vv) for k, vv in adj.items()}
                
                # Remove (u, v)
                new_adj[u].remove(v)
                new_adj[v].remove(u)
                
                # Remove (a, b)
                new_adj[a].remove(b)
                new_adj[b].remove(a)
                
                # Add new node
                new_node_id = max(adj.keys()) + 1
                new_adj[new_node_id] = []
                
                # Connect
                new_adj[a].append(new_node_id)
                new_adj[new_node_id].append(a)
                
                new_adj[b].append(new_node_id)
                new_adj[new_node_id].append(b)
                
                # Connect to v (root of pruned subtree)
                new_adj[v].append(new_node_id)
                new_adj[new_node_id].append(v)
                
                # Calculate SSR
                new_ssr = _calculate_ssr(new_adj, leaf_map, dist_obs)
                
                if new_ssr < current_ssr - 1e-9:
                    adj = new_adj
                    current_ssr = new_ssr
                    improved = True
                    break # Restart search
            
            if improved:
                break
        
        if not improved:
            break
            
    return adj, current_ssr

def _adj_to_newick(adj, leaf_map, is_leaf):
    """Convert adjacency list to Newick string."""
    # Find a root (any internal node)
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        # All leaves? n=2 or 1
        if len(adj) == 1:
            return leaf_map[list(adj.keys())[0]] + ";"
        elif len(adj) == 2:
            n1, n2 = list(adj.keys())
            return f"({leaf_map[n1]},{leaf_map[n2]});"
    
    visited = set()
    
    def build(node, parent):
        visited.add(node)
        children = []
        for neighbor in adj[node]:
            if neighbor != parent:
                child_str = build(neighbor, node)
                children.append(child_str)
        
        if is_leaf.get(node, False):
            return leaf_map[node]
        else:
            return "(" + ",".join(children) + ")"
    
    newick = build(root, -1)
    return newick + ";"

def _neighbor_joining_newick(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
        r = d_active.sum(axis=1)
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 5:
        if n <= 2:
            return ",".join(labels) + ";"
        return "(" + ",".join(labels) + ");"
    
    # 1. NJ
    nj_newick = _neighbor_joining_newick(d, labels)
    
    # 2. Parse to adj
    adj, leaf_map, is_leaf = _parse_newick_to_adj(nj_newick, n)
    
    # 3. SPR Refine
    if time.time() - start_time < deadline - 20: # Allow some time for SPR
        adj, ssr = _spr_refine(adj, leaf_map, d, max_passes=5)
    else:
        ssr = _calculate_ssr(adj, leaf_map, d)
    
    # 4. Convert back to Newick
    final_newick = _adj_to_newick(adj, leaf_map, is_leaf)
    
    return final_newick
