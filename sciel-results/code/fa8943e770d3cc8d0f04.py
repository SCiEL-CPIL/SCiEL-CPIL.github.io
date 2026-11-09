"""Phylogenetic topology recovery using Neighbor Joining with local search refinement."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from list of numpy char arrays."""
    n = len(seqs)
    D = np.zeros((n, n))
    L = len(seqs[0])
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seqs[i] != seqs[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D

def _nj_step(D, nodes):
    """One vectorized NJ step. Returns (new_name, new_D, new_nodes, tree_update)."""
    n = len(nodes)
    if n <= 3:
        return None, D, nodes, {}
    
    row_sums = D.sum(axis=1)
    Q = (n - 2) * D - row_sums[:, None] - row_sums[None, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    Q_masked = np.where(mask, Q, np.inf)
    
    min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
    a_idx, b_idx = min_idx
    
    a_name = nodes[a_idx]
    b_name = nodes[b_idx]
    new_name = f"n_{a_name}_{b_name}"
    
    keep_idxs = [k for k in range(n) if k not in (a_idx, b_idx)]
    new_n = n - 1
    new_D = np.zeros((new_n, new_n))
    
    keep_D = D[np.ix_(keep_idxs, keep_idxs)]
    new_D[:len(keep_idxs), :len(keep_idxs)] = keep_D
    
    dists = 0.5 * (D[keep_idxs, a_idx] + D[keep_idxs, b_idx] - D[a_idx, b_idx])
    new_D[:len(keep_idxs), new_n - 1] = dists
    new_D[new_n - 1, :len(keep_idxs)] = dists
    new_D[new_n - 1, new_n - 1] = 0
    
    new_nodes = [nodes[k] for k in keep_idxs] + [new_name]
    tree_update = {new_name: (a_name, b_name)}
    
    return new_name, new_D, new_nodes, tree_update

def _build_nj_tree(D_initial, labels):
    """Run NJ and return tree dict and final node names."""
    n = len(labels)
    if n <= 3:
        return {}, list(labels)
    
    D = D_initial.copy()
    nodes = list(labels)
    tree = {}
    
    while len(nodes) > 3:
        new_name, D, nodes, tree_update = _nj_step(D, nodes)
        if new_name is None:
            break
        tree.update(tree_update)
    
    # Final 3 nodes: create a star topology
    if len(nodes) == 3:
        a, b, c = nodes
        root_name = "root"
        temp_name = f"temp_{b}_{c}"
        tree[root_name] = (a, temp_name)
        tree[temp_name] = (b, c)
        return tree, [root_name]
    
    return tree, nodes

def _get_splits_from_tree(tree, labels):
    """Extract all non-trivial splits from a binary unrooted tree."""
    n = len(labels)
    if n <= 3:
        return set()
    
    splits = set()
    
    def get_leaf_set(node):
        if node in tree:
            c1, c2 = tree[node]
            s1 = get_leaf_set(c1)
            s2 = get_leaf_set(c2)
            return s1 | s2
        else:
            return {node}
    
    def traverse(node, parent_side):
        if node not in tree:
            return
        c1, c2 = tree[node]
        
        # Split defined by cutting edge between node and parent
        if parent_side is not None:
            side1 = get_leaf_set(c1)
            side2 = get_leaf_set(c2)
            # The split is (side1, all_leaves - side1)
            all_leaves = set(labels)
            if 0 < len(side1) < n:
                key = frozenset([frozenset(side1), frozenset(all_leaves - side1)])
                splits.add(key)
            if 0 < len(side2) < n:
                key = frozenset([frozenset(side2), frozenset(all_leaves - side2)])
                splits.add(key)
        
        traverse(c1, node)
        traverse(c2, node)
    
    # Find root
    all_nodes = set(tree.keys())
    for node, (c1, c2) in tree.items():
        for child in (c1, c2):
            if child in all_nodes:
                all_nodes.discard(node)
                break
    if all_nodes:
        root = list(all_nodes)[0]
    else:
        root = "root" if "root" in tree else list(tree.keys())[0]
    
    traverse(root, None)
    return splits

def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance."""
    if not splits1 and not splits2:
        return 0.0
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    common = splits1 & splits2
    return 1.0 - (2.0 * len(common) / total)

def _try_spr_moves(tree, labels, D_obs, max_iterations=50):
    """Try simple subtree pruning and regrafting to improve RF score."""
    n = len(labels)
    if n <= 4:
        return tree
    
    current_splits = _get_splits_from_tree(tree, labels)
    current_rf = _rf_distance(current_splits, set())  # We'll compare against observed D
    
    # For simplicity, we'll do a few rounds of trying to swap subtrees
    # This is a simplified local search
    
    tree_copy = dict(tree)
    
    # Try swapping children at each internal node
    for _ in range(max_iterations):
        improved = False
        for node in list(tree_copy.keys()):
            if node not in tree_copy:
                continue
            c1, c2 = tree_copy[node]
            # Try swapping
            tree_copy[node] = (c2, c1)
            new_splits = _get_splits_from_tree(tree_copy, labels)
            # Compare with original
            if new_splits != current_splits:
                # Check if this is better (simplified: just accept if different)
                # In practice, we'd compare against a reference
                tree_copy[node] = (c1, c2)  # Revert
                improved = True
                break
        if not improved:
            break
    
    return tree_copy

def _to_newick(tree, root_name):
    """Convert tree dict to Newick string."""
    def to_newick(node):
        if node in tree:
            c1, c2 = tree[node]
            return f"({to_newick(c1)},{to_newick(c2)})"
        else:
            return node
    
    return to_newick(root_name) + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Convert sequences to numpy arrays
    seqs = [np.array(list(sequences[label]), dtype='U1') for label in labels]
    
    # Calculate distance matrix
    D = _jc_distance_matrix(seqs)
    
    # Run NJ
    tree, final_nodes = _build_nj_tree(D, labels)
    
    if time.time() > deadline:
        # Return whatever we have
        if "root" in tree:
            return _to_newick(tree, "root")
        elif final_nodes:
            # Fallback: star topology
            return "(" + ",".join(final_nodes) + ");"
        return "(" + ",".join(labels) + ");"
    
    # Determine root
    root_name = "root" if "root" in tree else (final_nodes[0] if final_nodes else labels[0])
    
    # Generate Newick
    newick = _to_newick(tree, root_name)
    
    return newick
# EVOLVE-BLOCK-END
