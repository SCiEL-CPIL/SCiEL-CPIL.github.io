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

def _quartet_ml_scores_vectorized(arr):
    """
    Vectorized quartet scoring. For each quartet (a,b,c,d), compute scores
    for the three topologies and return the winning topology and confidence.
    """
    n, L = arr.shape
    # Precompute pairwise distance matrix
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    p_matrix = diff_matrix / L
    p_safe = np.clip(p_matrix, 0, 0.7499)
    d_matrix = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d_matrix, 0)
    
    # For n up to 32, number of quartets is C(n,4) which is manageable
    # For n=32: C(32,4) = 359600 quartets
    # We process them in batches for efficiency
    
    quartet_topologies = np.zeros((n, n, n, n), dtype=np.int8)
    quartet_weights = np.zeros((n, n, n, n), dtype=np.float32)
    
    # Generate all quartets
    indices = np.arange(n)
    quartets = list(combinations(range(n), 4))
    
    # Process in batches
    batch_size = 10000
    for start in range(0, len(quartets), batch_size):
        batch = quartets[start:start+batch_size]
        if not batch:
            break
        
        # Check time
        if _time_left() < 5.0:
            break
            
        ba = np.array([q[0] for q in batch])
        bb = np.array([q[1] for q in batch])
        bc = np.array([q[2] for q in batch])
        bd = np.array([q[3] for q in batch])
        
        # Compute scores for all three topologies
        s1 = d_matrix[ba, bb] + d_matrix[bc, bd] - d_matrix[ba, bc] - d_matrix[bb, bd]
        s2 = d_matrix[ba, bc] + d_matrix[bb, bd] - d_matrix[ba, bb] - d_matrix[bc, bd]
        s3 = d_matrix[ba, bd] + d_matrix[bc, bb] - d_matrix[ba, bb] - d_matrix[bc, bd]
        
        scores = np.stack([s1, s2, s3], axis=-1)
        min_scores = scores.min(axis=-1)
        argmin = scores.argmin(axis=-1)
        sorted_scores = np.sort(scores, axis=-1)
        weights = np.maximum(sorted_scores[:, 1] - sorted_scores[:, 0], 0.0)
        
        for i, (a, b, c, d) in enumerate(batch):
            quartet_topologies[a, b, c, d] = argmin[i]
            quartet_weights[a, b, c, d] = weights[i]
    
    return quartet_topologies, quartet_weights, d_matrix

def _quartet_to_splits_vectorized(quartet_topologies, quartet_weights, n):
    """
    Convert quartet votes to weighted splits using vectorized operations.
    """
    split_weights = {}
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    topo = quartet_topologies[a, b, c, d]
                    weight = quartet_weights[a, b, c, d]
                    
                    if weight < 1e-6:
                        continue
                    
                    if topo == 0:
                        side1 = frozenset([a, b])
                    elif topo == 1:
                        side1 = frozenset([a, c])
                    else:
                        side1 = frozenset([a, d])
                    
                    all_leaves = frozenset(range(n))
                    side2 = all_leaves - side1
                    
                    # Normalize split representation
                    if len(side1) < len(side2):
                        split = side1
                    elif len(side2) < len(side1):
                        split = side2
                    else:
                        if tuple(sorted(side1)) < tuple(sorted(side2)):
                            split = side1
                        else:
                            split = side2
                    
                    split_weights[split] = split_weights.get(split, 0.0) + weight
    
    return split_weights

def _build_tree_from_weighted_splits(split_weights, n, labels, global_indices=None):
    """
    Build a tree from weighted splits using greedy compatible split selection.
    """
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    if global_indices is None:
        global_indices = list(range(n))
    
    global_set = frozenset(global_indices)
    
    # Filter to splits that partition the current set
    candidate_splits = []
    for split, weight in split_weights.items():
        side1 = split & global_set
        side2 = global_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            candidate_splits.append((side1, weight))
    
    if not candidate_splits:
        return "(" + ",".join(labels) + ")"
    
    # Sort by weight (descending)
    candidate_splits.sort(key=lambda x: -x[1])
    
    # Greedy compatible split selection
    chosen_splits = []
    for split, weight in candidate_splits:
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
    
    tree1 = _build_tree_from_weighted_splits(split_weights, len(side1_globals), labels1, side1_globals)
    tree2 = _build_tree_from_weighted_splits(split_weights, len(side2_globals), labels2, side2_globals)
    
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

def _dpp_refine(tree_str, arr, labels, n, L, max_iterations=50):
    """
    Refine tree topology using DPP score improvement via split exchanges.
    """
    current_splits = _get_splits_from_tree(tree_str, labels)
    current_score = _dpp_score_vectorized(current_splits, arr, n, L)
    
    if not current_splits or n < 4:
        return tree_str
    
    all_leaves = frozenset(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    best_tree = tree_str
    best_score = current_score
    
    for iteration in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        improved = False
        
        # Try adding each possible split and removing incompatible ones
        # Generate candidate splits from all pairs of leaves
        candidate_splits = set()
        for i in range(n):
            for j in range(i+1, n):
                side1 = frozenset([labels[i], labels[j]])
                if 1 < len(side1) < n:
                    side2 = all_leaves - side1
                    if 1 < len(side2) < n:
                        if tuple(sorted(side1)) < tuple(sorted(side2)):
                            candidate_splits.add(side1)
                        else:
                            candidate_splits.add(side2)
        
        # Try each candidate split
        for new_split in candidate_splits:
            # Check if this split is already present
            if new_split in current_splits:
                continue
            
            # Find all splits incompatible with new_split
            incompatible = set()
            for s in current_splits:
                s_side1 = s
                s_side2 = all_leaves - s
                ns_side1 = new_split
                ns_side2 = all_leaves - new_split
                
                if (s_side1 & ns_side1) and (s_side1 & ns_side2) and (s_side2 & ns_side1) and (s_side2 & ns_side2):
                    incompatible.add(s)
            
            if not incompatible:
                continue
            
            # Try removing incompatible splits and adding new_split
            trial_splits = (current_splits - incompatible) | {new_split}
            
            # Check if trial_splits forms a valid tree (all splits compatible)
            valid = True
            for s1 in trial_splits:
                for s2 in trial_splits:
                    if s1 == s2:
                        continue
                    s1_side1 = s1
                    s1_side2 = all_leaves - s1
                    s2_side1 = s2
                    s2_side2 = all_leaves - s2
                    
                    if (s1_side1 & s2_side1) and (s1_side1 & s2_side2) and (s1_side2 & s2_side1) and (s1_side2 & s2_side2):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Compute score
            trial_score = _dpp_score_vectorized(trial_splits, arr, n, L)
            
            if trial_score < best_score - 1e-6:
                best_score = trial_score
                current_splits = trial_splits
                improved = True
        
        if not improved:
            break
    
    # Rebuild tree from best splits
    if current_splits != _get_splits_from_tree(best_tree, labels):
        best_tree = _splits_to_tree(current_splits, labels)
    
    return best_tree

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

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_nj, splits_nj = _nj(d_full, labels)
    
    # Quartet-based approach with vectorized scoring
    quartet_topologies, quartet_weights, d_matrix = _quartet_ml_scores_vectorized(arr)
    split_weights = _quartet_to_splits_vectorized(quartet_topologies, quartet_weights, n)
    
    # Build tree from weighted splits
    tree_quartet = _build_tree_from_weighted_splits(split_weights, n, labels)
    
    # Compare DPP scores
    dpp_nj = _dpp_score_vectorized(splits_nj, arr, n, L)
    dpp_quartet = _dpp_score_vectorized(_get_splits_from_tree(tree_quartet, labels), arr, n, L)
    
    # Choose the better tree
    if dpp_quartet < dpp_nj:
        best_tree = tree_quartet
    else:
        best_tree = tree_nj
    
    # DPP refinement if time permits
    if _time_left() > 30.0:
        refined_tree = _dpp_refine(best_tree, arr, labels, n, L)
        dpp_refined = _dpp_score_vectorized(_get_splits_from_tree(refined_tree, labels), arr, n, L)
        dpp_best = _dpp_score_vectorized(_get_splits_from_tree(best_tree, labels), arr, n, L)
        if dpp_refined < dpp_best:
            best_tree = refined_tree
    
    return best_tree
# EVOLVE-BLOCK-END
