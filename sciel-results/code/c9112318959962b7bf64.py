# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.85
_TOTAL_BUDGET = 270.0

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TOTAL_BUDGET * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _quartet_ml_scores(arr):
    """
    For each quartet (a,b,c,d), compute the quartet distance score for the three
    possible unrooted topologies under Jukes-Cantor.
    
    The three topologies are:
      T1: (a,b)|(c,d) - a and b are sisters, c and d are sisters
      T2: (a,c)|(b,d) - a and c are sisters, b and d are sisters  
      T3: (a,d)|(b,c) - a and d are sisters, b and c are sisters
    
    Using the standard quartet distance score:
    For topology (a,b)|(c,d): S1 = d(a,b) + d(c,d) - d(a,c) - d(b,d)
    For topology (a,c)|(b,d): S2 = d(a,c) + d(b,d) - d(a,b) - d(c,d)
    For topology (a,d)|(b,c): S3 = d(a,d) + d(b,c) - d(a,b) - d(c,d)
    
    The topology with the most negative score is preferred.
    """
    n, L = arr.shape
    # Precompute pairwise differences
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    p_matrix = diff_matrix / L
    p_safe = np.clip(p_matrix, 0, 0.7499)
    d_matrix = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d_matrix, 0)
    
    # For each quartet, compute scores for the three topologies
    # The most negative score wins
    
    quartet_scores = {}  # key: (a,b,c,d) with a<b<c<d, value: (best_topology, weight)
    
    for a, b, c, d in combinations(range(n), 4):
        # Three possible pairings
        # T1: (a,b)|(c,d)
        s1 = d_matrix[a,b] + d_matrix[c,d] - d_matrix[a,c] - d_matrix[b,d]
        # T2: (a,c)|(b,d)
        s2 = d_matrix[a,c] + d_matrix[b,d] - d_matrix[a,b] - d_matrix[c,d]
        # T3: (a,d)|(b,c)
        s3 = d_matrix[a,d] + d_matrix[b,c] - d_matrix[a,b] - d_matrix[c,d]
        
        scores = [s1, s2, s3]
        min_score = min(scores)
        # Weight = difference between best and second best (confidence)
        sorted_scores = sorted(scores)
        weight = sorted_scores[1] - sorted_scores[0]  # >= 0
        
        if min_score == s1:
            topology = 0  # (a,b)|(c,d)
        elif min_score == s2:
            topology = 1  # (a,c)|(b,d)
        else:
            topology = 2  # (a,d)|(b,c)
        
        quartet_scores[(a,b,c,d)] = (topology, max(weight, 0.0))
    
    return quartet_scores, d_matrix

def _quartet_to_splits(quartet_scores, n):
    """
    Convert quartet votes to a set of splits.
    For each quartet, the winning topology defines a split.
    We count how many times each split is supported.
    """
    split_counts = {}
    
    for (a,b,c,d), (topology, weight) in quartet_scores.items():
        if topology == 0:
            # (a,b)|(c,d)
            side1 = frozenset([a, b])
        elif topology == 1:
            # (a,c)|(b,d)
            side1 = frozenset([a, c])
        else:
            # (a,d)|(b,c)
            side1 = frozenset([a, d])
        
        all_leaves = frozenset(range(n))
        side2 = all_leaves - side1
        
        # Normalize: use the smaller side, or the lexicographically smaller
        if len(side1) < len(side2):
            split = side1
        elif len(side2) < len(side1):
            split = side2
        else:
            if tuple(sorted(side1)) < tuple(sorted(side2)):
                split = side1
            else:
                split = side2
        
        split_counts[split] = split_counts.get(split, 0.0) + weight
    
    return split_counts

def _build_tree_from_weighted_splits(split_counts, n, labels, global_indices=None):
    """
    Build a tree from weighted splits.
    global_indices: if provided, maps local indices 0..n-1 to global indices.
    """
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Get the global indices for this subtree
    if global_indices is None:
        global_indices = list(range(n))
    
    global_set = frozenset(global_indices)
    
    # Filter to splits that partition the current set
    candidate_splits = []
    for split, weight in split_counts.items():
        side1 = split & global_set
        side2 = global_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            candidate_splits.append((side1, weight))
    
    if not candidate_splits:
        return "(" + ",".join(labels) + ")"
    
    # Find a compatible set of splits using greedy approach
    candidate_splits.sort(key=lambda x: -x[1])
    
    chosen_splits = []
    for split, weight in candidate_splits:
        # Check compatibility with all chosen splits
        compatible = True
        for existing in chosen_splits:
            e_side1 = existing
            e_side2 = global_set - existing
            s_side1 = split
            s_side2 = global_set - split
            
            # Two splits are incompatible if they cross
            if (e_side1 & s_side1) and (e_side1 & s_side2) and (e_side2 & s_side1) and (e_side2 & s_side2):
                compatible = False
                break
        
        if compatible:
            chosen_splits.append(split)
    
    if not chosen_splits:
        return "(" + ",".join(labels) + ")"
    
    # Use the most balanced split as the top-level split
    best_split = None
    best_balance = 0
    for split in chosen_splits:
        balance = min(len(split), n - len(split))
        if balance > best_balance:
            best_balance = balance
            best_split = split
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    # Build local index to global index mapping
    global_to_local = {g: i for i, g in enumerate(global_indices)}
    
    side1_globals = sorted(best_split, key=lambda x: global_to_local[x])
    side2_globals = sorted(global_set - best_split, key=lambda x: global_to_local[x])
    
    # Recursively build subtrees
    labels1 = [labels[global_to_local[i]] for i in side1_globals]
    labels2 = [labels[global_to_local[i]] for i in side2_globals]
    
    tree1 = _build_tree_from_weighted_splits(split_counts, len(side1_globals), labels1, side1_globals)
    tree2 = _build_tree_from_weighted_splits(split_counts, len(side2_globals), labels2, side2_globals)
    
    return f"({tree1},{tree2})"

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _get_splits_from_tree(tree_str, all_labels):
    """Extract splits from a Newick tree string."""
    labels_set = set(all_labels)
    n = len(labels_set)
    
    def parse_tree(s, pos=0):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse_tree(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                else:
                    break
            pos += 1
            return ('node', children), pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',;':
                end += 1
            label = s[pos:end].strip()
            return ('leaf', label), pos
    
    node, _ = parse_tree(tree_str)
    
    def get_leaf_set(n):
        if n[0] == 'leaf':
            return frozenset([n[1]])
        else:
            result = frozenset()
            for child in n[1]:
                result = result | get_leaf_set(child)
            return result
    
    splits = set()
    def traverse(n):
        if n[0] == 'node':
            leaf_set = get_leaf_set(n)
            if 1 < len(leaf_set) < n:
                side1 = leaf_set
                side2 = frozenset(labels_set - leaf_set)
                if tuple(sorted(side1)) < tuple(sorted(side2)):
                    splits.add(side1)
                else:
                    splits.add(side2)
            for child in n[1]:
                traverse(child)
    
    traverse(node)
    return splits

def _splits_to_tree(splits, labels):
    """Convert a set of splits to a Newick tree string."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(labels)
    n = len(labels)
    
    # Filter splits to only those that partition the current label set
    valid_splits = []
    for s in splits:
        side1 = s & label_set
        side2 = label_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            valid_splits.append(side1)
    
    if not valid_splits:
        return "(" + ",".join(labels) + ")"
    
    # Find the most balanced split
    best_split = None
    best_balance = 0
    for s in valid_splits:
        balance = min(len(s), n - len(s))
        if balance > best_balance:
            best_balance = balance
            best_split = s
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    side1 = sorted(best_split, key=lambda x: int(x[1:]))
    side2 = sorted(label_set - best_split, key=lambda x: int(x[1:]))
    
    tree1 = _splits_to_tree(splits, side1)
    tree2 = _splits_to_tree(splits, side2)
    
    return f"({tree1},{tree2})"

def _dpp_score_vectorized(splits, arr, n, L):
    """
    Compute DPP score using vectorized operations.
    Lower is better.
    """
    if not splits or n < 3:
        return 0.0
    
    # Precompute pairwise difference matrix
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    total_score = 0.0
    for split in splits:
        side1 = np.array([1 if l in split else 0 for l in range(n)])
        n1 = side1.sum()
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            continue
        
        # Sum of differences between groups
        cross_diff = diff_matrix[np.ix_(side1.astype(bool), ~side1.astype(bool))].sum()
        
        count_pairs = n1 * n2
        if count_pairs == 0:
            continue
        
        avg_p = cross_diff / (count_pairs * L)
        avg_p = min(avg_p, 0.7499)
        
        d = -0.75 * np.log(1 - (4.0/3.0) * avg_p)
        total_score += (n1 * n2 / L) * d
    
    return total_score

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_full, splits_full = _nj(d_full, labels)
    
    # Quartet-based approach
    quartet_scores, d_matrix = _quartet_ml_scores(arr)
    split_counts = _quartet_to_splits(quartet_scores, n)
    
    # Build tree from weighted splits
    tree_quartet = _build_tree_from_weighted_splits(split_counts, n, labels)
    
    # Compare DPP scores
    dpp_nj = _dpp_score_vectorized(splits_full, arr, n, L)
    dpp_quartet = _dpp_score_vectorized(_get_splits_from_tree(tree_quartet, labels), arr, n, L)
    
    if dpp_quartet < dpp_nj:
        return tree_quartet
    else:
        return tree_full
# EVOLVE-BLOCK-END
