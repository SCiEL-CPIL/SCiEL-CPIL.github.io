import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])

    # Encode sequences: use ASCII values directly for comparison
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr

    # Vectorized pairwise difference counting
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs

    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _split_compatible(split1, split2, n):
    """Check if two splits are compatible (can coexist in a tree)."""
    # Two splits A|B and C|D are compatible iff one of A∩C, A∩D, B∩C, B∩D is empty
    # where A∪B = C∪D = all taxa
    a = split1
    b = set(range(n)) - a
    c = split2
    d = set(range(n)) - c
    
    return len(a & c) == 0 or len(a & d) == 0 or len(b & c) == 0 or len(b & d) == 0


def _split_weight(D, split, n):
    """
    Compute the weight of a split using the distance matrix.
    A split is well-supported if it partitions the taxa into groups
    that are internally close and externally far.
    """
    a = list(split)
    b = [i for i in range(n) if i not in split]
    
    if len(a) == 0 or len(b) == 0:
        return 0.0
    
    # Average distance within each group
    if len(a) > 1:
        within_a = D[np.ix_(a, a)].sum() / (len(a) * (len(a) - 1))
    else:
        within_a = 0.0
    
    if len(b) > 1:
        within_b = D[np.ix_(b, b)].sum() / (len(b) * (len(b) - 1))
    else:
        within_b = 0.0
    
    # Average distance between groups
    between = D[np.ix_(a, b)].mean()
    
    # Weight: high between-group distance, low within-group distances
    # This is related to the "split support" in split decomposition
    weight = between - 0.5 * (within_a + within_b)
    return max(0.0, weight)


def _extract_splits_from_NJ(D, labels, n):
    """
    Extract all splits from the NJ construction process.
    Returns a list of (split_frozenset, weight) pairs.
    """
    splits = []
    
    # We'll track the splits as we build the NJ tree
    # This requires modifying NJ to record splits at each step
    
    D_work = D.copy()
    node_names = list(range(n))
    
    while len(node_names) > 3:
        m = len(node_names)
        r = D_work.sum(axis=1)
        Q = (m - 2) * D_work - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        # The new node represents the split of {i,j} vs the rest
        # In terms of original taxa, we need to track which original taxa
        # are in each current node
        
        # For simplicity, we'll use a different approach:
        # After building the NJ tree, extract splits from it
        break
    
    return splits


def _tree_to_splits(tree_dict, n):
    """
    Convert a tree represented as a nested dict to its set of splits.
    Each internal edge defines a split.
    """
    splits = set()
    all_taxa = set(range(n))
    
    def get_leaves(node):
        if 'leaf' in node and node['leaf'] is not None:
            return {node['leaf']}
        leaves = set()
        for child in node.get('children', []):
            leaves |= get_leaves(child)
        return leaves
    
    def traverse(node):
        leaves = get_leaves(node)
        if len(leaves) > 1 and len(leaves) < n:
            # Normalize: use the smaller side
            if len(leaves) <= n / 2:
                splits.add(frozenset(leaves))
            else:
                splits.add(frozenset(all_taxa - leaves))
        for child in node.get('children', []):
            traverse(child)
    
    traverse(tree_dict)
    return splits


def _build_tree_from_splits(splits, n):
    """
    Build a tree from a set of compatible splits.
    Uses a recursive approach: find a split that partitions the taxa,
    then recursively build subtrees.
    """
    if n <= 1:
        return {'leaf': 0}
    if n == 2:
        return {'children': [{'leaf': 0}, {'leaf': 1}]}
    if n == 3:
        return {'children': [{'leaf': 0}, {'leaf': 1}, {'leaf': 2}]}
    
    # Find a split that partitions the taxa
    # Prefer splits with size around n/2 for balanced trees
    best_split = None
    best_score = -1
    
    for split in splits:
        s = len(split)
        if 1 < s < n - 1:
            # Score: prefer splits close to n/2
            score = -abs(s - n / 2)
            if score > best_score:
                best_score = score
                best_split = split
    
    if best_split is None:
        # No good split found, create a star tree
        return {'children': [{'leaf': i} for i in range(n)]}
    
    a = list(best_split)
    b = [i for i in range(n) if i not in best_split]
    
    # Recursively build subtrees
    # We need to filter splits to those relevant to each subtree
    splits_a = set()
    splits_b = set()
    for split in splits:
        sa = split & best_split
        sb = split - best_split
        if len(sa) > 1 and len(sa) < len(a):
            splits_a.add(sa)
        if len(sb) > 1 and len(sb) < len(b):
            splits_b.add(sb)
    
    # Build subtrees with relabeled taxa
    def build_subtree(taxa, sub_splits):
        if len(taxa) <= 3:
            return {'children': [{'leaf': i} for i in taxa]}
        
        # Find best split within this subtree
        best_sub_split = None
        best_sub_score = -1
        for split in sub_splits:
            s = len(split)
            if 1 < s < len(taxa) - 1:
                score = -abs(s - len(taxa) / 2)
                if score > best_sub_score:
                    best_sub_score = score
                    best_sub_split = split
        
        if best_sub_split is None:
            return {'children': [{'leaf': i} for i in taxa]}
        
        a_sub = list(best_sub_split)
        b_sub = [i for i in taxa if i not in best_sub_split]
        
        splits_a_sub = set()
        splits_b_sub = set()
        for split in sub_splits:
            sa = split & best_sub_split
            sb = split - best_sub_split
            if len(sa) > 1 and len(sa) < len(a_sub):
                splits_a_sub.add(sa)
            if len(sb) > 1 and len(sb) < len(b_sub):
                splits_b_sub.add(sb)
        
        return {'children': [build_subtree(a_sub, splits_a_sub),
                             build_subtree(b_sub, splits_b_sub)]}
    
    subtree_a = build_subtree(a, splits_a)
    subtree_b = build_subtree(b, splits_b)
    
    return {'children': [subtree_a, subtree_b]}


def _tree_to_newick(node, label_map):
    """Convert tree dict to Newick string."""
    if 'leaf' in node and node['leaf'] is not None:
        return label_map[node['leaf']]
    children = node.get('children', [])
    if len(children) == 1:
        return f"({_tree_to_newick(children[0], label_map)})"
    return "(" + ",".join(_tree_to_newick(c, label_map) for c in children) + ")"


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D (numpy array, symmetric, zero diag).
    labels: list of taxon names corresponding to rows/cols of D.
    Returns a Newick string (no trailing semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)

    while len(node_names) > 3:
        m = len(node_names)
        # Compute row sums
        r = D.sum(axis=1)  # shape (m,)

        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        # Set diagonal to inf
        np.fill_diagonal(Q, np.inf)
        # Find min
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"

        # Compute new distances
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Build new distance matrix using fancy indexing
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1

        # Extract submatrix for kept nodes
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]

        # Add new node distances
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]

        D = new_D
        node_names = new_names

    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _build_tree_from_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns a nested dict structure with 'name', 'children', 'length' fields.
    """
    def parse(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            # Check for branch length
            if pos < len(s) and s[pos] != ',' and s[pos] != ')':
                end = pos
                while end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                    end += 1
                length_str = s[pos:end].strip()
                try:
                    length = float(length_str)
                except ValueError:
                    length = 0.0
                pos = end
            else:
                length = 0.0
            return {'name': None, 'children': children, 'length': length}, pos
        else:
            end = pos
            while end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                end += 1
            name = s[pos:end].strip()
            # Check for branch length
            if end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                length_end = end
                while length_end < len(s) and s[length_end] != ',' and s[length_end] != ')' and s[length_end] != ';':
                    length_end += 1
                length_str = s[end:length_end].strip()
                try:
                    length = float(length_str)
                except ValueError:
                    length = 0.0
                end = length_end
            else:
                length = 0.0
            return {'name': name, 'children': [], 'length': length}, end
    
    tree, _ = parse(newick, 0)
    return tree


def _get_leaves(node, leaf_set=None):
    """Get all leaf names in a subtree."""
    if leaf_set is None:
        leaf_set = set()
    if node['name'] is not None:
        leaf_set.add(node['name'])
    for child in node['children']:
        _get_leaves(child, leaf_set)
    return leaf_set


def _tree_to_splits_newick(tree, all_leaves):
    """
    Convert a tree (from Newick parse) to its set of non-trivial splits.
    A split is a frozenset of one side of an internal edge.
    We represent each split by the frozenset of leaves on one side (choose the smaller side for uniqueness).
    """
    splits = set()
    
    def add_split(leaf_set):
        # Normalize: use the smaller side (or either if equal)
        if len(leaf_set) <= len(all_leaves) / 2:
            split = frozenset(leaf_set)
        else:
            split = frozenset(all_leaves - leaf_set)
        # Only add non-trivial splits (size > 1 and < n-1)
        if 1 < len(split) < len(all_leaves) - 1:
            splits.add(split)
    
    def traverse(node, parent_side):
        """
        For each edge, determine the split it defines.
        parent_side is the set of leaves on the parent side of this edge.
        """
        if node['name'] is not None:
            # Leaf node
            return
        
        for child in node['children']:
            child_leaves = _get_leaves(child)
            # The split defined by this edge: child_leaves vs all_leaves - child_leaves
            add_split(child_leaves)
            traverse(child, all_leaves - child_leaves)
    
    traverse(tree, all_leaves)
    return splits


def _newick_to_string(node):
    """Convert tree structure back to Newick string."""
    if node['name'] is not None:
        return node['name']
    if len(node['children']) == 1:
        child_str = _newick_to_string(node['children'][0])
        if node['length'] > 0:
            return f"({child_str}){node['length']:.4f}"
        return f"({child_str})"
    child_strs = [_newick_to_string(c) for c in node['children']]
    inner = ",".join(child_strs)
    if node['length'] > 0:
        return f"({inner}){node['length']:.4f}"
    return f"({inner})"


def _find_max_compatible_splits(splits_with_weights, n):
    """
    Find a maximum-weight set of compatible splits.
    Uses a greedy approach: sort by weight, add if compatible with all selected.
    """
    # Sort by weight descending
    sorted_splits = sorted(splits_with_weights, key=lambda x: x[1], reverse=True)
    
    selected = []
    selected_splits = set()
    
    for split, weight in sorted_splits:
        if weight <= 0:
            continue
        # Check compatibility with all selected splits
        compatible = True
        for sel_split in selected_splits:
            if not _split_compatible(split, sel_split, n):
                compatible = False
                break
        if compatible:
            selected.append((split, weight))
            selected_splits.add(split)
    
    return selected


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Fallback: star tree
        return "(" + ",".join(labels) + ";"

    # Strategy 1: NJ as baseline
    nj_newick = _neighbor_joining(D, labels)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return nj_newick + ";"

    # Strategy 2: Split-based approach
    # Generate candidate splits from various sources
    candidate_splits = []
    
    # Source 1: Splits from NJ tree
    nj_tree = _build_tree_from_newick(nj_newick)
    all_leaves = set(labels)
    nj_splits = _tree_to_splits_newick(nj_tree, all_leaves)
    
    # Create label-to-index mapping
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Convert NJ splits to index-based
    for split in nj_splits:
        idx_split = frozenset(label_to_idx[lab] for lab in split)
        weight = _split_weight(D, idx_split, n)
        candidate_splits.append((idx_split, weight))
    
    # Source 2: Generate additional splits from distance matrix
    # Try all 2|n-2 and 3|n-3 splits (for small n)
    if n <= 12:
        from itertools import combinations
        # 2|n-2 splits
        for pair in combinations(range(n), 2):
            split = frozenset(pair)
            weight = _split_weight(D, split, n)
            candidate_splits.append((split, weight))
        
        # 3|n-3 splits
        for triple in combinations(range(n), 3):
            split = frozenset(triple)
            weight = _split_weight(D, split, n)
            candidate_splits.append((split, weight))
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return nj_newick + ";"
    
    # Find maximum compatible set of splits
    max_splits = _find_max_compatible_splits(candidate_splits, n)
    
    if len(max_splits) > 0:
        # Build tree from selected splits
        split_set = set(split for split, _ in max_splits)
        tree_dict = _build_tree_from_splits(split_set, n)
        label_map = {i: labels[i] for i in range(n)}
        split_newick = _tree_to_newick(tree_dict, label_map)
        
        # Compare with NJ: prefer the one with more splits (better resolved)
        # Count non-trivial splits in each
        nj_tree2 = _build_tree_from_newick(nj_newick)
        nj_split_count = len(_tree_to_splits_newick(nj_tree2, all_leaves))
        
        # Parse split tree and count its splits
        split_tree_parsed = _build_tree_from_newick(split_newick)
        split_split_count = len(_tree_to_splits_newick(split_tree_parsed, all_leaves))
        
        # Prefer the tree with more splits (more resolved = potentially better RF)
        if split_split_count >= nj_split_count:
            result = split_newick
        else:
            result = nj_newick
    else:
        result = nj_newick

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 2:
        return result + ";"

    # Ensure all taxa appear exactly once
    return result + ";"
