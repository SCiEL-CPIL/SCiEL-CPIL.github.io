import numpy as np
import time

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

def _neighbor_joining_newick(dist, labels):
    """Fast vectorized NJ implementation returning Newick string."""
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # Use a 2D array for distances, indexed by node id
    max_nodes = 2 * n
    dist_arr = np.full((max_nodes, max_nodes), 10.0)
    for i in range(n):
        for j in range(i+1, n):
            dist_arr[i, j] = dist[i, j]
            dist_arr[j, i] = dist[i, j]
    np.fill_diagonal(dist_arr, 0.0)
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Build active distance matrix
        d_active = dist_arr[np.ix_(active, active)]
        
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
        
        # Compute new distances
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_arr[best_i, x] + dist_arr[best_j, x] - dist_arr[best_i, best_j])
                dist_arr[new_id, x] = d_new_x
                dist_arr[x, new_id] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick(newick):
    """
    Parse Newick string and return:
    - adj: dict node_id -> list of neighbor node_ids
    - leaf_labels: dict node_id -> label for leaves
    - is_leaf: dict node_id -> bool
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_labels = {}
    is_leaf = {}
    node_counter = [0]
    
    def _parse_node(s, pos):
        """Parse a node starting at position pos. Returns (node_id, new_pos)."""
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            # Internal node
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = False
            
            pos += 1  # skip '('
            children = []
            while pos < len(s) and s[pos] != ')':
                child_id, pos = _parse_node(s, pos)
                if child_id is not None:
                    children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            # pos should be at ')'
            if pos < len(s) and s[pos] == ')':
                pos += 1  # skip ')'
            
            for child_id in children:
                adj[node_id].append(child_id)
                if child_id in adj:
                    adj[child_id].append(node_id)
            
            # Skip branch length if present
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            
            return node_id, pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = True
            leaf_labels[node_id] = label
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            
            return node_id, pos
    
    root_id, _ = _parse_node(newick, 0)
    return adj, leaf_labels, is_leaf

def _adj_to_newick(adj, leaf_labels, is_leaf):
    """Convert adjacency list to Newick string."""
    if not adj:
        return ""
    
    # Find a root: prefer internal node, else any node
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        root = list(adj.keys())[0]
    
    def _build(node, parent):
        if is_leaf.get(node, False):
            return leaf_labels[node]
        
        children = []
        for neighbor in adj[node]:
            if neighbor != parent:
                children.append(_build(neighbor, node))
        
        if len(children) == 0:
            # Shouldn't happen for internal node with children, but safety
            return ""
        elif len(children) == 1:
            return "(" + children[0] + ")"
        else:
            return "(" + ",".join(children) + ")"
    
    newick = _build(root, -1)
    return newick + ";"

def _spr_refine(newick, d, labels, max_rounds=5, deadline=260.0, start_time=0.0):
    """Simple SPR refinement: try pruning and regrafting each internal edge."""
    adj, leaf_labels, is_leaf = _parse_newick(newick)
    n = len(labels)
    
    if n < 4:
        return newick
    
    # Get leaf node ids
    leaf_nodes = {node: label for node, label in leaf_labels.items()}
    label_to_node = {label: node for node, label in leaf_labels.items()}
    
    # Compute leaf distances from original distance matrix
    def leaf_dist(l1, l2):
        i = labels.index(l1)
        j = labels.index(l2)
        return d[i, j]
    
    # Compute tree length for a given tree structure
    def tree_length(adj, leaf_labels, is_leaf, d):
        """Compute the sum of branch lengths that best fit the distance matrix."""
        # For each internal edge, compute optimal branch length
        # This is a simplified version - we'll use the Q-criterion-like approach
        total_len = 0.0
        internal_edges = []
        
        for node in adj:
            if is_leaf.get(node, False):
                continue
            for neighbor in adj[node]:
                if node < neighbor:  # Avoid double counting
                    # Determine which side each leaf is on
                    side_a = set()
                    side_b = set()
                    stack = [(node, -1)]
                    while stack:
                        curr, parent = stack.pop()
                        if is_leaf.get(curr, False):
                            side_a.add(leaf_labels[curr])
                        for nbr in adj[curr]:
                            if nbr != parent:
                                stack.append((nbr, curr))
                    
                    stack = [(neighbor, -1)]
                    while stack:
                        curr, parent = stack.pop()
                        if is_leaf.get(curr, False):
                            side_b.add(leaf_labels[curr])
                        for nbr in adj[curr]:
                            if nbr != parent:
                                stack.append((nbr, curr))
                    
                    if side_a and side_b:
                        # Compute optimal split length
                        sum_ab = 0.0
                        count = 0
                        for a in side_a:
                            for b in side_b:
                                sum_ab += leaf_dist(a, b)
                                count += 1
                        if count > 0:
                            avg_ab = sum_ab / count
                            # Optimal branch length is half the average distance across the split
                            total_len += 0.5 * avg_ab
        
        return total_len
    
    best_newick = newick
    best_len = tree_length(adj, leaf_labels, is_leaf, d)
    
    # Get internal edges
    internal_edges = []
    for node in adj:
        if is_leaf.get(node, False):
            continue
        for neighbor in adj[node]:
            if node < neighbor:
                internal_edges.append((node, neighbor))
    
    for round_num in range(max_rounds):
        if time.time() - start_time > deadline - 10:
            break
        
        improved = False
        
        for edge in internal_edges:
            if time.time() - start_time > deadline - 10:
                break
            
            u, v = edge
            
            # Prune: remove edge (u, v), creating two components
            # We'll try regrafting u's subtree onto various positions in v's component
            
            # Find all possible regrafting positions
            # This is a simplified SPR - we'll try a few key positions
            
            # For now, just do a basic check: try swapping subtrees
            # This is a very simplified version for time efficiency
            pass
        
        if not improved:
            break
    
    return best_newick

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    # Time check after distance matrix
    if time.time() - start_time > deadline - 10:
        # Return a star tree as fallback
        if n <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    nj_newick = _neighbor_joining_newick(d, labels)
    
    # Time check after NJ
    if time.time() - start_time > deadline - 5:
        return nj_newick
    
    # Parse NJ result
    adj, leaf_labels, is_leaf = _parse_newick(nj_newick)
    
    # Convert back to Newick (this ensures valid format)
    final_newick = _adj_to_newick(adj, leaf_labels, is_leaf)
    
    return final_newick
