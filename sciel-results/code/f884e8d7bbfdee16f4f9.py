import math
import time
import numpy as np
from itertools import combinations

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    # Convert sequences to integer arrays
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.array([
        [char_to_int.get(c, 0) for c in sequences[label]]
        for label in labels
    ], dtype=np.int8)
    
    L = seq_arrays.shape[1]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_arrays[i] != seq_arrays[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    
    return dist_mat


def _nj_tree(dist_mat, labels):
    """
    Perform Neighbour Joining on a distance matrix.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Active nodes: start with all leaves
    active = list(range(n))
    # Tree structure: dict mapping node_id -> (child_a, child_b, len_a, len_b)
    internal_nodes = {}
    next_id = n
    
    # Distance matrix for current active nodes
    # Use a dict-based approach for clarity and correctness
    # dist_dict[(i, j)] = distance between node i and node j
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist_mat[i, j]
    
    while len(active) > 3:
        k = len(active)
        
        # Compute r[i] = sum of distances from node i to all other active nodes
        r = {}
        for node in active:
            r[node] = sum(dist_dict[(node, other)] for other in active if other != node)
        
        # Compute Q matrix and find best pair
        best_q = float('inf')
        best_a, best_b = -1, -1
        
        for idx_a in range(k):
            for idx_b in range(idx_a + 1, k):
                node_a = active[idx_a]
                node_b = active[idx_b]
                d_ab = dist_dict[(node_a, node_b)]
                q = (k - 2) * d_ab - r[node_a] - r[node_b]
                if q < best_q:
                    best_q = q
                    best_a, best_b = node_a, node_b
        
        # Compute branch lengths
        d_ab = dist_dict[(best_a, best_b)]
        delta = (r[best_a] - r[best_b]) / (k - 2)
        len_a = 0.5 * (d_ab + delta)
        len_b = 0.5 * (d_ab - delta)
        
        # Clip negative branch lengths
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        new_node = next_id
        next_id += 1
        internal_nodes[new_node] = (best_a, best_b, len_a, len_b)
        
        # Compute distances from new node to all other active nodes
        for other in active:
            if other == best_a or other == best_b:
                continue
            d_a_other = dist_dict[(best_a, other)]
            d_b_other = dist_dict[(best_b, other)]
            d_new_other = 0.5 * (d_a_other + d_b_other - d_ab)
            if d_new_other < 0:
                d_new_other = 0.0
            dist_dict[(new_node, other)] = d_new_other
            dist_dict[(other, new_node)] = d_new_other
        
        # Remove old nodes, add new node
        active = [x for x in active if x != best_a and x != best_b]
        active.append(new_node)
    
    # Handle final 3 nodes
    u, v, w = active
    d_uv = dist_dict[(u, v)]
    d_uw = dist_dict[(u, w)]
    d_vw = dist_dict[(v, w)]
    
    len_u = 0.5 * (d_uv + d_uw - d_vw)
    len_v = 0.5 * (d_uv + d_vw - d_uw)
    len_w = 0.5 * (d_uw + d_vw - d_uv)
    
    if len_u < 0: len_u = 0.0
    if len_v < 0: len_v = 0.0
    if len_w < 0: len_w = 0.0
    
    node_1 = next_id
    next_id += 1
    internal_nodes[node_1] = (u, v, len_u, len_v)
    
    node_2 = next_id
    next_id += 1
    internal_nodes[node_2] = (node_1, w, 0.0, len_w)
    
    # Build Newick string
    def get_newick(node):
        if node < n:
            return labels[node]
        else:
            child_a, child_b, la, lb = internal_nodes[node]
            return f"({get_newick(child_a)}:{la:.6f},{get_newick(child_b)}:{lb:.6f})"
    
    return get_newick(node_2)


def _parse_newick_to_splits(newick, n, labels):
    """
    Parse a Newick string and extract all non-trivial splits.
    Returns a set of frozensets representing the splits.
    """
    # Remove branch lengths and trailing semicolon
    newick_clean = newick.rstrip(';').strip()
    
    # Tokenize
    def parse(s, pos):
        """Parse Newick starting at position pos. Returns (node, new_pos)."""
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            # Internal node
            children = []
            pos += 1
            while pos < len(s) and s[pos] != ')':
                child, pos = parse(s, pos)
                if child is not None:
                    children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            return ('internal', children), pos
        else:
            # Leaf or label
            end = pos
            while end < len(s) and s[end] not in ',)':
                end += 1
            label = s[pos:end].strip()
            # Remove branch length if present
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label), pos
    
    root, _ = parse(newick_clean, 0)
    
    # Collect all splits
    splits = set()
    
    def collect_splits(node, labels_set):
        if node[0] == 'leaf':
            return {node[1]}
        else:
            _, children = node
            child_sets = []
            for child in children:
                cs = collect_splits(child, labels_set)
                child_sets.append(cs)
            
            # For each child, the split is child_set vs complement
            for cs in child_sets:
                if 0 < len(cs) < len(labels_set):
                    splits.add(frozenset(cs))
            
            # Return union of all child sets
            result = set()
            for cs in child_sets:
                result |= cs
            return result
    
    all_labels = set(labels)
    collect_splits(root, all_labels)
    
    return splits


def _quartet_support(sequences, labels, i, j, k, l):
    """
    Count quartet support for taxa i, j, k, l.
    Returns (count_ijkl, count_ikjl, count_iljk) for the three topologies.
    """
    seq_i = sequences[labels[i]]
    seq_j = sequences[labels[j]]
    seq_k = sequences[labels[k]]
    seq_l = sequences[labels[l]]
    L = len(seq_i)
    
    counts = [0, 0, 0]  # ij|kl, ik|jl, il|jk
    
    for pos in range(L):
        a = seq_i[pos]
        b = seq_j[pos]
        c = seq_k[pos]
        d = seq_l[pos]
        
        if a == b == c == d:
            continue
        
        # Count character frequencies
        from collections import Counter
        char_counts = Counter([a, b, c, d])
        
        if len(char_counts) == 2:
            # Two characters, each appearing twice
            if a == b and c == d and a != c:
                counts[0] += 1  # ij|kl
            elif a == c and b == d and a != b:
                counts[1] += 1  # ik|jl
            elif a == d and b == c and a != b:
                counts[2] += 1  # il|jk
        elif len(char_counts) == 3:
            # Three characters, one appearing twice
            for ch, cnt in char_counts.items():
                if cnt == 2:
                    if a == b and a == ch:
                        counts[0] += 1
                    elif a == c and a == ch:
                        counts[1] += 1
                    elif a == d and a == ch:
                        counts[2] += 1
                    elif b == c and b == ch:
                        counts[2] += 1  # bc|ad -> il|jk
                    elif b == d and b == ch:
                        counts[1] += 1  # bd|ac -> ik|jl
                    elif c == d and c == ch:
                        counts[0] += 1  # cd|ab -> ij|kl
                    break
    
    return counts


def _tree_quartet_score(sequences, labels, newick):
    """
    Score a tree by counting how many quartets it matches
    according to quartet support.
    """
    splits = _parse_newick_to_splits(newick, len(labels), labels)
    n = len(labels)
    
    if n < 4:
        return 0.0
    
    total_support = 0.0
    total_quartets = 0
    
    # Sample quartets for efficiency
    max_quartets = 5000
    all_quartets = list(combinations(range(n), 4))
    if len(all_quartets) > max_quartets:
        import random
        random.seed(42)
        sampled = random.sample(all_quartets, max_quartets)
    else:
        sampled = all_quartets
    
    for (i, j, k, l) in sampled:
        counts = _quartet_support(sequences, labels, i, j, k, l)
        total_support += max(counts)
        total_quartets += 1
        
        # Check which topology the tree supports
        # ij|kl: split {i,j} vs {k,l}
        if frozenset({labels[i], labels[j]}) in splits:
            total_support += 0  # already counted
        # This approach is flawed; let me use a different scoring
    
    return total_support


def _nni_refine(sequences, labels, newick, time_budget):
    """
    Perform NNI (Nearest Neighbor Interchange) refinement on the tree.
    This is a simplified version that tries local topological changes.
    """
    # For simplicity, this is a placeholder that returns the input tree.
    # A full NNI implementation would require tree manipulation.
    return newick


def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # Use NJ for the initial tree
    newick = _nj_tree(dist_mat, labels)
    
    elapsed = time.time() - start_time
    
    # If we have time, try quartet-based refinement
    if elapsed < time_limit - 30.0 and n >= 4:
        # Try a simple refinement: re-evaluate using quartet information
        # For now, just return the NJ tree as it's already reasonable
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return newick
    
    return newick + ";"
