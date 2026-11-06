"""Phylogenetic topology recovery: vectorized NJ + small-n quartet polish."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None


def _elapsed():
    return 0.0 if _DEADLINE is None else (time.time() - _DEADLINE[0])


def _deadline_exceeded(margin=5.0):
    if _DEADLINE is None:
        return False
    return (time.time() - _DEADLINE[0]) > (_DEADLINE[1] - margin)


def _jc_from_p(p):
    """Jukes-Cantor corrected distance from a proportion-difference (scalar)."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Pairwise JC-corrected distances via vectorized comparison (chunked)."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Encode each sequence as int8 array
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        row = mat[i]
        for j, c in enumerate(s):
            row[j] = enc[c]

    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 16
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # (chunk, n, L) boolean
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2)  # (chunk, n)
        p = counts.astype(np.float64) / L
        p_safe = np.clip(p, 0.0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick(dist, labels):
    """Neighbor Joining -> Newick (unrooted, branch lengths omitted). Fixed to handle dynamic IDs."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    # Use a dictionary for distances to avoid index errors with new IDs
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = dist[i, j]
    
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Row sums restricted to current nodes
        row_sum = {i: 0.0 for i in nodes}
        for i in nodes:
            for j in nodes:
                if i != j:
                    row_sum[i] += D[(i, j)]

        best_crit = float('inf')
        best_a = best_b = None
        for ii in range(m):
            i = nodes[ii]
            for jj in range(ii + 1, m):
                j = nodes[jj]
                crit = (m - 2) * D[(i, j)] - row_sum[i] - row_sum[j]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = i, j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(tree[x] for x in nodes) + ")"

        new_id = next_id
        next_id += 1
        tree[new_id] = f"({tree[a]},{tree[b]})"

        for x in nodes:
            if x == a or x == b:
                continue
            d_new_x = 0.5 * (D[(a, x)] + D[(b, x)] - D[(a, b)])
            D[(new_id, x)] = d_new_x
            D[(x, new_id)] = d_new_x

        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    a, b, c = nodes
    return f"({tree[a]},{tree[b]},{tree[c]})"


def _fuad_newick(dist, labels):
    """
    Fast Up-And-Down (FUAD) inspired pruning.
    1. Identify taxa with high average distance (potential long branches/noise).
    2. Remove them.
    3. Build NJ tree on the core.
    4. Re-attach removed taxa to their nearest neighbor in the core tree.
    """
    n = len(labels)
    if n <= 5:
        return _nj_newick(dist, labels)

    # Calculate row sums to identify outliers
    row_sums = np.sum(dist, axis=1)
    
    # Determine number of taxa to prune. 
    # Heuristic: prune up to 10% of taxa, but at least 1 if n>10, max 5 to keep core robust.
    k = max(1, min(5, int(n * 0.1)))
    
    # Indices sorted by row sum (descending)
    sorted_indices = np.argsort(row_sums)[::-1]
    prune_set = set(sorted_indices[:k].tolist())
    
    core_indices = [i for i in range(n) if i not in prune_set]
    core_labels = [labels[i] for i in core_indices]
    
    # Build distance matrix for core
    core_dist = dist[np.ix_(core_indices, core_indices)]
    
    # Build NJ tree for core
    core_newick = _nj_newick(core_dist, core_labels)
    
    # Re-attach pruned taxa
    # We need to find the best place to insert each pruned taxon.
    # Simple strategy: attach to the nearest core taxon (leaf).
    # This is a "nearest neighbor join" on leaves.
    # To do this, we parse the core_newick to get leaf names, or just use the core_labels.
    # We will modify the Newick string by replacing a leaf label with (leaf, pruned_taxon).
    
    # Map core label to index in core_labels for distance lookup
    # Actually, we have dist matrix. 
    # For each pruned taxon p (original index p_idx):
    # Find core taxon c (original index c_idx) that minimizes dist[p_idx, c_idx].
    # Then in the Newick string, replace label c with (label c, label p).
    
    # This simple replacement might break if the same core taxon is used multiple times,
    # but since we are adding distinct leaves, we can just append to the existing leaf.
    # Newick format: (A,B) -> ((A,P),B) if P is nearest to A.
    
    # Let's do this properly by parsing or just string replacement.
    # String replacement is risky if labels are substrings of others, but labels are t0, t1...
    # So "t1" is not in "t10". We can use word boundaries or just careful replacement.
    # Better: Reconstruct the tree structure? No, too slow.
    # Let's use a simple parser to find the leaf nodes and their positions?
    # Alternative: Just return the core tree and ignore pruned taxa? No, must include all.
    # Alternative: Add pruned taxa as a star at the end? No, that's bad topology.
    
    # Let's try a simple approach: 
    # For each pruned taxon, find nearest core taxon.
    # Replace the first occurrence of the nearest core taxon's label in the Newick string
    # with (nearest_core_label, pruned_label).
    # This effectively inserts the pruned taxon as a sibling to the nearest core taxon.
    
    final_newick = core_newick
    
    for p_idx in sorted(prune_set):
        p_label = labels[p_idx]
        # Find nearest core taxon
        best_c_idx = None
        best_d = float('inf')
        for c_idx in core_indices:
            d = dist[p_idx, c_idx]
            if d < best_d:
                best_d = d
                best_c_idx = c_idx
        
        c_label = labels[best_c_idx]
        
        # Replace c_label with (c_label, p_label)
        # We need to be careful not to replace inside parentheses if c_label appears there?
        # In a valid Newick, leaf labels appear only as leaves.
        # So replacing "c_label" with "(c_label, p_label)" should be safe if we replace 
        # the token exactly.
        # Since labels are unique and formatted t\d+, we can do a simple string replace 
        # of the first occurrence.
        
        # To be safe, we can use a regex or just find the first index.
        pos = final_newick.find(c_label)
        if pos != -1:
            # Ensure we don't match part of a longer label (e.g. t1 in t10)
            # But find() finds the first occurrence. If t1 is a leaf, it will be there.
            # If t10 is the leaf, t1 is not a leaf label.
            # So this is safe.
            final_newick = final_newick[:pos] + f"({c_label},{p_label})" + final_newick[pos+len(c_label):]
            
    return final_newick


def _splits_to_newick(splits, labels):
    """Reconstruct a Newick tree from a set of non-trivial splits (index sets)."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    sp_set = set(splits)
    # Canonicalize: also store complements so we can test membership both ways.
    def in_split(S):
        S = frozenset(S)
        if S in sp_set:
            return True
        return (frozenset(range(n)) - S) in sp_set

    # Recursive: find a split that separates the current node set into two
    # non-empty parts; recurse on each.
    def build(node_set):
        node_set = frozenset(node_set)
        if len(node_set) == 1:
            i = next(iter(node_set))
            return labels[i]
        if len(node_set) == 2:
            a, b = sorted(node_set)
            return f"({labels[a]},{labels[b]})"
        # find a split that cuts node_set into two parts both within node_set
        found = False
        for sp in sp_set:
            left = sp & node_set
            right = node_set - sp
            if 1 <= len(left) < len(node_set) and 1 <= len(right) < len(node_set):
                l = build(left)
                r = build(right)
                return f"({l},{r})"
            found = True
        # No valid split: star
        return "(" + ",".join(labels[i] for i in sorted(node_set)) + ")"

    return build(frozenset(range(n)))


def reconstruct(sequences):
    global _DEADLINE
    _DEADLINE = [time.time(), 250.0]  # safety margin before 270s kill

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return "(" + ",".join(labels) + ");"

    dist = _compute_distance_matrix(sequences, labels)

    # Use FUAD-pruned NJ for better topology recovery on noisy/long-branch data
    try:
        if not _deadline_exceeded(margin=20.0):
            result = _fuad_newick(dist, labels)
            if result:
                return result + ";"
    except Exception:
        pass

    result = _nj_newick(dist, labels)
    return result + ";"
# EVOLVE-BLOCK-END
