import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255  # Safety margin before 270s kill

def _check_deadline():
    return time.time() < _DEADLINE

def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    # Vectorized distance computation
    # a and b are strings
    diff = np.sum(np.array([x != y for x, y in zip(a, b)], dtype=np.int8))
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    """Compute JC distance matrix using NumPy."""
    n_taxa = len(labels)
    # Convert sequences to integer arrays
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seq_array = np.zeros((n_taxa, n_sites), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Map characters to ints
        # Using np.vectorize or lookup table
        # Faster: translate string to bytes, then map
        # But simple loop is okay for N=32, S=2000
        for j in range(n_sites):
            seq_array[i, j] = state_map[seq[j]]
    
    # Compute pairwise differences
    # diff[i, j] = number of sites where seq_i != seq_j
    # We can do this by broadcasting
    # seq_array is (N, S)
    # We want sum over S of (seq_array[i] != seq_array[j])
    # This is an N x N x S boolean array, which is 32*32*2000 = 2M bools. 2MB. Fine.
    diff_matrix = np.zeros((n_taxa, n_taxa), dtype=np.int32)
    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            diff = np.sum(seq_array[i] != seq_array[j])
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff
    
    # Convert to JC distances
    p_matrix = diff_matrix.astype(np.float64) / n_sites
    # Handle p >= 0.75
    p_matrix[p_matrix >= 0.7499] = 0.7499
    dist_matrix = -0.75 * np.log(1.0 - 4.0 * p_matrix / 3.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    return dist_matrix, seq_array

def _nj_newick_fast(dist_matrix, labels):
    """Vectorized Neighbor-Joining."""
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    # Current nodes in the matrix
    # We will maintain a list of active nodes
    # and a mapping from node index to Newick string
    active = list(range(n))
    newick_strs = {i: labels[i] for i in range(n)}
    
    # dist is a copy of the distance matrix
    d = dist_matrix.copy()
    
    while len(active) > 3:
        if not _check_deadline():
            break
            
        k = len(active)
        # Compute r[i] = sum_{j in active, j!=i} d[i, j]
        # r is a vector of length k
        # We need to map active indices to matrix indices
        active_indices = np.array(active)
        
        # Extract submatrix for active nodes
        d_sub = d[np.ix_(active_indices, active_indices)]
        
        # r[i] = sum of row i, excluding diagonal (which is 0)
        r = np.sum(d_sub, axis=1)
        
        # Compute Q matrix: Q[i, j] = (k-2)*d[i, j] - r[i] - r[j]
        # We want to find the pair (i, j) with minimum Q
        # Q is symmetric
        Q = (k - 2) * d_sub - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, 0) # Diagonal is 0, we don't care
        
        # Find min Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        i_node = active[i_idx]
        j_node = active[j_idx]
        
        # Create new node
        new_node = len(newick_strs)
        newick_strs[new_node] = f"({newick_strs[i_node]},{newick_strs[j_node]})"
        
        # Update distance matrix
        # d[new, x] = 0.5 * (d[i, x] + d[j, x] - d[i, j])
        # for all x in active, x != i, j
        
        # We need to add a row/col to d
        d_new = np.zeros((d.shape[0] + 1, d.shape[0] + 1), dtype=np.float64)
        d_new[:d.shape[0], :d.shape[0]] = d
        d_new[new_node, :d.shape[0]] = 0.0
        d_new[:d.shape[0], new_node] = 0.0
        
        for x in active:
            if x == i_node or x == j_node:
                continue
            d_new[new_node, x] = 0.5 * (d[i_node, x] + d[j_node, x] - d[i_node, j_node])
            d_new[x, new_node] = d_new[new_node, x]
        
        d = d_new
        
        # Update active list
        active.remove(i_node)
        active.remove(j_node)
        active.append(new_node)
        
    # Final Newick
    # Join the remaining 3 nodes
    # They are in 'active'
    final_nodes = [newick_strs[x] for x in active]
    # Sort to ensure deterministic output? Not required, but good for debugging
    return "(" + ",".join(final_nodes) + ");"

def _parse_newick(newick):
    """Parse a Newick string into a nested tuple structure."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    def parse(s):
        if s[0] == '(':
            depth = 0
            i = 1
            parts = []
            current = ''
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                    current += s[i]
                elif s[i] == ')':
                    depth -= 1
                    if depth == -1:
                        break
                    current += s[i]
                elif s[i] == ',' and depth == 0:
                    parts.append(current)
                    current = ''
                else:
                    current += s[i]
                i += 1
            if i < len(s):
                i += 1
            rest = s[i:]
            if rest.startswith(':'):
                j = 0
                while j < len(rest) and rest[j] not in (',', '(', ')', ';'):
                    j += 1
            children = [parse(p) for p in parts]
            return ('internal', children)
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label)
    
    return parse(newick)

def _build_adjacency(tree):
    """Build adjacency list from tree structure."""
    adj = {}
    node_types = {}
    counter = [0]
    
    def assign_id(node, parent_id=None):
        if node[0] == 'leaf':
            nid = ('leaf', node[1])
            node_types[nid] = ('leaf', node[1])
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            return nid
        else:
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            for child in node[1]:
                cid = assign_id(child, nid)
            return nid
    
    root = assign_id(tree)
    node_ids = list(adj.keys())
    return adj, node_ids, node_types

def _root_tree(adj, node_ids, node_types):
    """Root the tree and build parent/children structure."""
    root = None
    for nid in node_ids:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = node_ids[0]
    
    parent = {root: None}
    children = {nid: [] for nid in node_ids}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)
    
    return root, parent, children

def _postorder(root, children):
    """Get postorder traversal."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order

def _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, base_length):
    """Compute ML log-likelihood using vectorized NumPy operations."""
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    exp_term = math.exp(-4.0 * base_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    
    prob = {}
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            p = np.zeros((n_sites, 4), dtype=np.float64)
            obs = seq_array[idx]
            p[np.arange(n_sites), obs] = 1.0
            prob[nid] = p
        else:
            prob[nid] = None
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            continue
        
        child_list = children[nid]
        if not child_list:
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            cp = prob[child]
            row_sum = cp.sum(axis=1, keepdims=True)
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll

def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure."""
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj

def _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, base_length, max_iterations=50):
    """Perform NNI local search to improve tree topology using ML scoring."""
    current_ll = _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, base_length)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_ll = current_ll
        best_adj = None
        
        for u in node_ids:
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u:
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a = u_nbrs[0]
                b = u_nbrs[1]
                c = v_nbrs[0]
                d = v_nbrs[1]
                
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                ll1 = _ml_score_tree_fast(new_adj1, node_ids, node_types, seq_array, n_sites, base_length)
                if ll1 > best_ll + 1e-6:
                    best_ll = ll1
                    best_adj = new_adj1
                
                new_adj2 = _apply_nni(adj, node_types, u, v, a, b, c, d, 2)
                ll2 = _ml_score_tree_fast(new_adj2, node_ids, node_types, seq_array, n_sites, base_length)
                if ll2 > best_ll + 1e-6:
                    best_ll = ll2
                    best_adj = new_adj2
        
        if best_adj is not None:
            adj = best_adj
            current_ll = best_ll
            improved = True
    
    return adj

def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in sorted(adj[nid], key=lambda x: str(x)):
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"

def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute distance matrix and seq array
    dist_matrix, seq_array = _compute_distance_matrix(sequences, labels)
    n_sites = len(sequences[labels[0]])
    
    # Step 2: Neighbor-Joining to get initial tree
    newick = _nj_newick_fast(dist_matrix, labels)
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Estimate branch length for ML
    total_d = np.sum(dist_matrix) / 2
    count = n_taxa * (n_taxa - 1) / 2
    mean_d = total_d / max(1, count)
    base_length = max(0.01, min(0.5, mean_d / max(1, n_taxa - 2)))
    
    # Step 4: NNI local search
    if _check_deadline():
        adj = _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, base_length, max_iterations=50)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate
    for label in labels:
        if label not in result:
            return newick
    
    return result
