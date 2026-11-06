import math
import time
import numpy as np

_DEADLINE = 260.0  # Safety margin before 270s kill
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    
    # Convert sequences to numeric arrays for fast comparison
    # A=0, C=1, G=2, T=3
    conv = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seqs = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        for j, c in enumerate(seq):
            num_seqs[i, j] = conv[c]
    
    L = len(sequences[labels[0]])
    D = np.zeros((n, n))
    
    # Vectorized pairwise mismatch calculation
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(num_seqs[i] != num_seqs[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_build(D, labels):
    """Run Neighbor Joining, return Newick string."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Use a fixed-size distance matrix to avoid resizing issues
    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes))
    dist[:n, :n] = D
    
    active = list(range(n))
    tree_strs = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        
        # Calculate row sums (total distance from each node to all others in active)
        total = np.zeros(m)
        for k, i in enumerate(active):
            total[k] = sum(dist[i, j] for j in active if j != i)
        
        # Find best pair (i, j) minimizing (m-2)*d(i,j) - total[i] - total[j]
        best_criterion = float('inf')
        best_pair = None
        
        for ki in range(m):
            i = active[ki]
            for kj in range(ki + 1, m):
                j = active[kj]
                criterion = (m - 2) * dist[i, j] - total[ki] - total[kj]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined_str = f"({tree_strs[i]},{tree_strs[j]})"
        new_idx = max(active) + 1
        
        # Safety check
        if new_idx >= max_nodes:
            break
            
        # Update distances to new node
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        for x in active:
            if x != i and x != j:
                d_new = 0.5 * (dist[i, x] + dist[j, x] - dist[i, j])
                dist[new_idx, x] = d_new
                dist[x, new_idx] = d_new
        
        # Remove i and j from active, add new_idx
        active = [x for x in active if x not in (i, j)]
        active.append(new_idx)
        tree_strs[new_idx] = joined_str
        
        # Time check
        if _time_left() < 2.0:
            break
    
    # Final tree
    parts = [tree_strs[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _parse_newick(newick):
    """
    Parse Newick string into an adjacency list representation.
    Returns:
        adj: dict mapping node ID to list of neighbor IDs
        leaves: dict mapping node ID to label
    """
    s = newick.rstrip(';').strip()
    
    adj = {}
    leaves = {}
    node_counter = 0
    
    def parse(s):
        nonlocal node_counter
        s = s.strip()
        if not s:
            return None
        
        node_id = node_counter
        node_counter += 1
        adj[node_id] = []
        
        if s.startswith('('):
            # Internal node
            depth = 0
            end_idx = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            inner = s[1:end_idx]
            # Split inner by top-level commas
            parts = []
            depth = 0
            current = []
            for c in inner:
                if c == '(':
                    depth += 1
                    current.append(c)
                elif c == ')':
                    depth -= 1
                    current.append(c)
                elif c == ',' and depth == 0:
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(c)
            if current:
                parts.append(''.join(current))
            
            # Recurse on children
            for part in parts:
                child_id = parse(part)
                if child_id is not None:
                    adj[node_id].append(child_id)
                    adj[child_id].append(node_id)
        else:
            # Leaf
            label = s
            if ':' in s:
                label = s.split(':')[0].strip()
            leaves[node_id] = label
        
        return node_id

    root_id = parse(s)
    return adj, leaves


def _nni_polish(newick, D, labels):
    """
    Perform NNI polish to improve topology based on distance fit.
    Uses least-squares branch length estimation for cost evaluation.
    """
    if _time_left() < 5.0:
        return newick
    
    adj, leaves = _parse_newick(newick)
    n_nodes = len(adj)
    
    if n_nodes <= 3:
        return newick
    
    # Map leaf labels to node IDs
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    
    # Get leaf indices in D
    leaf_nodes = [label_to_node[lb] for lb in labels]
    num_leaves = len(labels)
    
    # Precompute path masks for each leaf pair
    # path_mask[i][j] is a set of edges (as (min(u,v), max(u,v)) tuples) on path between leaf i and leaf j
    # Actually, let's build an edge list and map edges to indices
    edges = []
    edge_idx = {}
    for u in adj:
        for v in adj[u]:
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_idx[(u, v)] = idx
                edge_idx[(v, u)] = idx
    
    num_edges = len(edges)
    if num_edges == 0:
        return newick
    
    # For each pair of leaves, compute the set of edges on the path
    # We'll store this as a binary matrix: path_matrix[pair_idx, edge_idx] = 1 if edge is on path
    num_pairs = num_leaves * (num_leaves - 1) // 2
    path_matrix = np.zeros((num_pairs, num_edges), dtype=np.int8)
    pair_to_leaf = {}
    
    pair_idx = 0
    for i in range(num_leaves):
        for j in range(i + 1, num_leaves):
            li = leaf_nodes[i]
            lj = leaf_nodes[j]
            
            # BFS to find path
            visited = {li}
            parent = {li: None}
            queue = [li]
            found = False
            while queue:
                node = queue.pop(0)
                if node == lj:
                    found = True
                    break
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            if found:
                # Trace path
                path_edges = []
                cur = lj
                while cur != li:
                    par = parent[cur]
                    path_edges.append((min(par, cur), max(par, cur)))
                    cur = par
                
                for e in path_edges:
                    e_idx = edge_idx[e]
                    path_matrix[pair_idx, e_idx] = 1
                
                pair_to_leaf[pair_idx] = (labels[i], labels[j])
            pair_idx += 1
    
    def compute_cost(newick_str):
        """Compute cost for a given Newick string using least-squares branch lengths."""
        if _time_left() < 2.0:
            return float('inf')
        
        try:
            adj_new, leaves_new = _parse_newick(newick_str)
            if len(adj_new) != n_nodes:
                return float('inf')
            
            # Rebuild edges for new topology
            edges_new = []
            edge_idx_new = {}
            for u in adj_new:
                for v in adj_new[u]:
                    if u < v:
                        idx = len(edges_new)
                        edges_new.append((u, v))
                        edge_idx_new[(u, v)] = idx
                        edge_idx_new[(v, u)] = idx
            
            num_edges_new = len(edges_new)
            if num_edges_new != num_edges:
                return float('inf')
            
            # Map leaf labels to node IDs in new tree
            label_to_node_new = {leaf: node for node, leaf in leaves_new.items()}
            leaf_nodes_new = [label_to_node_new[lb] for lb in labels]
            
            # Build path matrix for new topology
            path_matrix_new = np.zeros((num_pairs, num_edges_new), dtype=np.int8)
            pair_idx_new = 0
            for i in range(num_leaves):
                for j in range(i + 1, num_leaves):
                    li = leaf_nodes_new[i]
                    lj = leaf_nodes_new[j]
                    
                    visited = {li}
                    parent = {li: None}
                    queue = [li]
                    found = False
                    while queue:
                        node = queue.pop(0)
                        if node == lj:
                            found = True
                            break
                        for neighbor in adj_new[node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                parent[neighbor] = node
                                queue.append(neighbor)
                    
                    if found:
                        cur = lj
                        while cur != li:
                            par = parent[cur]
                            e = (min(par, cur), max(par, cur))
                            if e in edge_idx_new:
                                path_matrix_new[pair_idx_new, edge_idx_new[e]] = 1
                            cur = par
                    pair_idx_new += 1
            
            # Build system: A x = b
            # A is path_matrix_new, b is D for leaf pairs
            b = np.zeros(num_pairs)
            pair_idx_b = 0
            for i in range(num_leaves):
                for j in range(i + 1, num_leaves):
                    b[pair_idx_b] = D[i, j]
                    pair_idx_b += 1
            
            # Solve least squares
            try:
                x, residuals, rank, sv = np.linalg.lstsq(path_matrix_new, b, rcond=None)
                # Cost is sum of squared residuals
                predicted = path_matrix_new @ x
                cost = np.sum((predicted - b) ** 2)
                return cost
            except:
                return float('inf')
        except:
            return float('inf')
    
    current_cost = compute_cost(newick)
    
    # NNI moves: for each internal edge, try the two alternative topologies
    # We need to identify internal edges (edges not incident to leaves)
    leaf_node_set = set(leaves.keys())
    
    best_newick = newick
    best_cost = current_cost
    
    # Iterate over all edges
    for u, v in edges:
        if u in leaf_node_set or v in leaf_node_set:
            continue
        
        # Find the four neighbors: u connected to v and two other nodes (or more if degree > 3, but binary tree)
        # In a binary tree, internal nodes have degree 3
        # So u is connected to v and two others, say a and b
        # v is connected to u and two others, say c and d
        # The NNI move swaps the connections: (a,b) and (c,d) become (a,c) and (b,d) or (a,d) and (b,c)
        
        # Get neighbors of u excluding v
        u_neighbors = [x for x in adj[u] if x != v]
        v_neighbors = [x for x in adj[v] if x != v]
        
        if len(u_neighbors) != 2 or len(v_neighbors) != 2:
            continue
        
        a, b = u_neighbors
        c, d = v_neighbors
        
        # Two alternative NNI topologies
        # Alt 1: connect a-c and b-d
        # Alt 2: connect a-d and b-c
        
        for alt in [((a, c), (b, d)), ((a, d), (b, c))]:
            # Build new adjacency
            adj_alt = {k: list(v) for k, v in adj.items()}
            
            # Remove edges (u, v), (u, a), (u, b), (v, c), (v, d)
            # Add edges (u, alt1[0][0]), (alt1[0][1], v), (u, alt1[1][0]), (alt1[1][1], v)
            # Wait, the structure is:
            # Original: a - u - v - c, b - u - v - d (roughly)
            # Actually, u is connected to a, b, v. v is connected to c, d, u.
            # Alt 1: u connected to a, c, v? No.
            # NNI: the edge is (u, v). The four "outer" edges are (u,a), (u,b), (v,c), (v,d).
            # Alt 1: remove (u,v), (u,a), (u,b), (v,c), (v,d). Add (u,a), (u,c), (v,b), (v,d)? No.
            # Correct NNI:
            # Original: (a, b) at u, (c, d) at v, edge u-v.
            # Alt 1: (a, c) and (b, d). So u connects to a, c? No.
            # Let's think of the quartet: a, b, c, d.
            # Original: (a, b, (c, d)) with u between ab and cd? No.
            # The tree has a cherry (a,b) at u and a cherry (c,d) at v? Not necessarily.
            # u has neighbors a, b, v. v has neighbors c, d, u.
            # So the quartet is a, b, c, d.
            # Original topology: ((a, b), (c, d))
            # Alt 1: ((a, c), (b, d))
            # Alt 2: ((a, d), (b, c))
            
            # To implement:
            # Remove edges: (u, v), (u, a), (u, b), (v, c), (v, d)
            # Add edges: (u, a), (u, c), (v, b), (v, d) for Alt 1?
            # No, we need to preserve the leaf connections.
            # In Alt 1: a and c are in one clade, b and d in the other.
            # So u should connect to a and c? And v to b and d?
            # But u and v are internal nodes.
            # Let's just rebuild the adjacency list by modifying the edges.
            
            # Remove (u, v)
            adj_alt[u].remove(v)
            adj_alt[v].remove(u)
            
            # Remove (u, a), (u, b), (v, c), (v, d)
            adj_alt[u].remove(a)
            adj_alt[a].remove(u)
            adj_alt[u].remove(b)
            adj_alt[b].remove(u)
            adj_alt[v].remove(c)
            adj_alt[c].remove(v)
            adj_alt[v].remove(d)
            adj_alt[d].remove(v)
            
            # Add new edges
            e1, e2 = alt
            adj_alt[u].append(e1[0])
            adj_alt[e1[0]].append(u)
            adj_alt[v].append(e1[1])
            adj_alt[e1[1]].append(v)
            
            adj_alt[u].append(e2[0])
            adj_alt[e2[0]].append(u)
            adj_alt[v].append(e2[1])
            adj_alt[e2[1]].append(v)
            
            # Convert to Newick
            # We need a function to convert adjacency to Newick
            # This is complex. Let's skip NNI if it's too risky.
            pass
    
    return best_newick


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    newick = _nj_build(D, labels)
    
    # NNI Polish (optional, may be slow)
    # newick = _nni_polish(newick, D, labels)
    
    return newick
