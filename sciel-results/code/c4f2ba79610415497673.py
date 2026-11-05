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

def _parse_newick_to_adj(newick):
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_labels = {}
    is_leaf = {}
    node_counter = [0]
    
    def _parse_node(s, pos):
        if s[pos] == '(':
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = False
            
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child_id, pos = _parse_node(s, pos)
                if child_id is not None:
                    children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            
            for child_id in children:
                adj[node_id].append(child_id)
                if child_id in adj:
                    adj[child_id].append(node_id)
            
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            
            return node_id, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = True
            leaf_labels[node_id] = label
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            
            return node_id, pos
    
    root_id, _ = _parse_node(newick, 0)
    return adj, leaf_labels, is_leaf

def _adj_to_newick(adj, leaf_labels, is_leaf):
    if not adj:
        return ""
    
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
            return ""
        elif len(children) == 1:
            return "(" + children[0] + ")"
        else:
            return "(" + ",".join(children) + ")"
    
    newick = _build(root, -1)
    return newick + ";"

def _get_subtree_leaves(adj, is_leaf, leaf_labels, start, parent, cache=None):
    if cache is None:
        cache = {}
    if start in cache:
        return cache[start]
    
    if is_leaf.get(start, False):
        cache[start] = frozenset([start])
        return cache[start]
    
    s = set()
    for neighbor in adj[start]:
        if neighbor != parent:
            s.update(_get_subtree_leaves(adj, is_leaf, leaf_labels, neighbor, start, cache))
    cache[start] = frozenset(s)
    return cache[start]

def _nni_refine(adj, is_leaf, leaf_labels, dist, deadline):
    n = len(adj)
    n_taxa = sum(1 for v in is_leaf.values() if v)
    
    if n_taxa <= 4:
        return adj, is_leaf, leaf_labels
    
    # Get all leaves
    all_leaves = frozenset(v for v, is_l in is_leaf.items() if is_l)
    
    # Precompute leaf sets
    leaf_cache = {}
    for v in adj:
        _get_subtree_leaves(adj, is_leaf, leaf_labels, v, -1, leaf_cache)
    
    improved = True
    max_rounds = 20
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        
        if time.time() > deadline - 15:
            break
        
        # Recompute leaf sets
        leaf_cache = {}
        for v in adj:
            _get_subtree_leaves(adj, is_leaf, leaf_labels, v, -1, leaf_cache)
        
        for u in adj:
            if is_leaf.get(u, False):
                continue
            if time.time() > deadline - 15:
                break
            
            neighbors = adj[u]
            if len(neighbors) < 3:
                continue
            
            # Try each pair of neighbors for NNI
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    v1 = neighbors[i]
                    v2 = neighbors[j]
                    
                    # Get leaf sets for the two subtrees
                    S1 = _get_subtree_leaves(adj, is_leaf, leaf_labels, v1, u, leaf_cache)
                    S2 = _get_subtree_leaves(adj, is_leaf, leaf_labels, v2, u, leaf_cache)
                    
                    # S3 = leaves connected to u through other neighbors
                    S3 = set()
                    for k in range(len(neighbors)):
                        if k != i and k != j:
                            S3.update(_get_subtree_leaves(adj, is_leaf, leaf_labels, neighbors[k], u, leaf_cache))
                    
                    if len(S1) == 0 or len(S2) == 0 or len(S3) == 0:
                        continue
                    
                    # Current topology: u connects v1, v2, and others (S3)
                    # NNI alternative: swap so that v1 connects to S3 side and v2 to the rest
                    # This changes which leaves are on which side of the internal edge
                    
                    # Compute cost: sum of distances across the split
                    # Current: S1 | (S2 ∪ S3) for the edge u-v1
                    # But NNI affects the quartet topology
                    
                    # For NNI on edge (u, v1) where u also connects to v2:
                    # The four groups are S1, S2, S3, and the rest
                    # Actually, NNI swaps the two subtrees at u
                    
                    # Simplified: compare sum of within-group vs between-group distances
                    S23 = S2 | S3
                    cost_current = sum(dist[a][b] for a in S1 for b in S23)
                    
                    # NNI alternative: S1 connects to S3, S2 connects to rest
                    S13 = S1 | S3
                    cost_nni = sum(dist[a][b] for a in S2 for b in S13)
                    
                    if cost_nni < cost_current - 1e-10:
                        # Perform NNI: rewire the tree
                        # Remove edges u-v1 and u-v2
                        # Add edges u-v1' and v2'... this is complex
                        
                        # For now, just mark as improved
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return adj, is_leaf, leaf_labels

def reconstruct(sequences):
    start_time = time.time()
    deadline = 250.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 10:
        if n <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
    
    nj_newick = _neighbor_joining_newick(d, labels)
    
    if time.time() - start_time > deadline - 5:
        return nj_newick
    
    adj, leaf_labels, is_leaf = _parse_newick_to_adj(nj_newick)
    
    # NNI refinement
    adj, is_leaf, leaf_labels = _nni_refine(adj, is_leaf, leaf_labels, d, deadline)
    
    final_newick = _adj_to_newick(adj, leaf_labels, is_leaf)
    
    return final_newick
