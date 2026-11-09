"""Phylogenetic topology recovery via BIONJ-corrected Neighbor-Joining with bootstrap voting and quartet refinement."""
import time
import math
import numpy as np
import re

_DEADLINE_FRACTION = 0.90
_start_time = None
_instance_count = 0


def _encode_sequences(sequences, labels):
    """Encode sequences as uint8 array for fast operations."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    return seqs


def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from encoded sequences."""
    n = seqs.shape[0]
    L = seqs.shape[1]
    
    # Pairwise differences using broadcasting
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    # JC correction
    pmat_clipped = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat_clipped / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _bionj_correct(dmat, pmat, n, L):
    """Apply BIONJ variance correction to distance matrix."""
    # Variance estimate per node
    # Under JC model, Var(p_ij) ≈ p_ij*(1-p_ij)/L approximately
    # BIONJ correction: d'_ij = d_ij - (σ_i + σ_j) / (2*(n-2))
    # where σ_i = sum_j p_ij^2 / L (simplified)
    
    # More accurate: variance of pairwise distance estimator
    # For each pair, Var(d_hat) ≈ (4/9) * (1/(1-4p/3)^2) * p(1-p)/L
    # Simplified: use pmat^2 sum as proxy
    
    var = (pmat ** 2).sum(axis=1) / max(L, 1)
    correction = (var[:, np.newaxis] + var[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    
    return dmat_corrected


def _neighbor_joining_vectorized(dmat, labels):
    """
    Perform Neighbor-Joining using numpy arrays for speed.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Use array-based NJ
    # active_indices: indices into the current distance matrix
    # current_dmat: full distance matrix (will be modified)
    # node_labels: mapping from index to label
    
    active = list(range(n))
    dmat_work = dmat.copy()
    node_labels = list(labels)
    tree_nodes = {}
    next_id = n
    
    while len(active) > 2:
        m = len(active)
        if m < 3:
            break
        
        # Get submatrix for active nodes
        sub = dmat_work[np.ix_(active, active)]
        
        # Compute row sums
        r = sub.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (m-2)*d[i,j] - r[i] - r[j]
        Q = (m - 2) * sub - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        i_node = active[i_idx]
        j_node = active[j_idx]
        
        d_ij = dmat_work[i_node, j_node]
        li = 0.5 * d_ij + (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        lj = 0.5 * d_ij - (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        
        # Create new node
        w_id = next_id
        next_id += 1
        w_label = f"w{w_id}"
        
        tree_nodes[w_label] = (node_labels[i_node], node_labels[j_node], max(li, 0.0), max(lj, 0.0))
        
        # Update distances
        for k_idx, k_node in enumerate(active):
            if k_node == i_node or k_node == j_node:
                continue
            d_wk = 0.5 * (dmat_work[i_node, k_node] + dmat_work[j_node, k_node] - d_ij)
            # Store in a temporary structure
            if w_id < dmat_work.shape[0]:
                dmat_work[w_id, k_node] = d_wk
                dmat_work[k_node, w_id] = d_wk
        
        # Remove i, j from active; add w
        active.remove(i_node)
        active.remove(j_node)
        active.append(w_id)
        
        # Extend dmat_work if needed
        if w_id >= dmat_work.shape[0]:
            new_size = w_id + 1
            new_dmat = np.zeros((new_size, new_size))
            old_size = dmat_work.shape[0]
            new_dmat[:old_size, :old_size] = dmat_work
            dmat_work = new_dmat
        
        # Update node_labels
        if w_id < len(node_labels):
            node_labels[w_id] = w_label
        else:
            node_labels.append(w_label)
    
    # Connect last two
    if len(active) == 2:
        u, v = active
        d_uv = dmat_work[u, v]
        w_id = next_id
        next_id += 1
        w_label = f"w{w_id}"
        tree_nodes[w_label] = (node_labels[u], node_labels[v], max(d_uv / 2.0, 0.0), max(d_uv / 2.0, 0.0))
        active = [w_id]
        if w_id < len(node_labels):
            node_labels[w_id] = w_label
        else:
            node_labels.append(w_label)
    
    # Build Newick
    leaf_set = set(labels)
    
    def build_newick(node_label):
        if node_label in leaf_set:
            return node_label
        if node_label not in tree_nodes:
            return node_label
        child1, child2, len1, len2 = tree_nodes[node_label]
        n1 = build_newick(child1)
        n2 = build_newick(child2)
        return f"({n1}:{len1:.4f},{n2}:{len2:.4f})"
    
    root = active[0] if active else None
    if root is not None and root < len(node_labels):
        return build_newick(node_labels[root])
    
    # Fallback
    return _star_tree(labels)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _extract_splits(newick, labels):
    """Parse Newick and extract unrooted splits as frozensets of tuples."""
    # Remove branch lengths and semicolons
    clean = re.sub(r':[0-9.]+', '', newick)
    clean = clean.replace(';', '').strip()
    
    labels_set = set(labels)
    splits = set()
    
    def parse(s, idx):
        if idx >= len(s):
            return set(), idx
        
        if s[idx] != '(':
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
        
        idx += 1
        children = []
        while idx < len(s) and s[idx] != ')':
            child, idx = parse(s, idx)
            children.append(child)
            if idx < len(s) and s[idx] == ',':
                idx += 1
        
        if idx < len(s) and s[idx] == ')':
            idx += 1
        
        combined = set()
        for child in children:
            combined.update(child)
        
        for child in children:
            if 0 < len(child) < len(labels):
                if len(child) <= len(labels) - len(child):
                    splits.add(frozenset(child))
                else:
                    other = labels_set - child
                    splits.add(frozenset(other))
        
        return combined, idx
    
    parse(clean, 0)
    return splits


def _topology_key(splits):
    """Create a hashable key from splits."""
    return frozenset(splits)


def _quartet_support(splits, labels):
    """Compute support for each quartet topology from the split set."""
    n = len(labels)
    if n < 4:
        return {}
    
    # For each 4-subset, determine which of the 3 topologies is supported
    # Topology 0: (ab|cd) - splits {a,b} and {c,d}
    # Topology 1: (ac|bd) - splits {a,c} and {b,d}  
    # Topology 2: (ad|bc) - splits {a,d} and {b,c}
    
    quartet_support = {}
    
    # Precompute all 4-subsets
    from itertools import combinations
    for quad in combinations(range(n), 4):
        a, b, c, d = [labels[i] for i in quad]
        
        # Check which topology is present
        top0 = frozenset({a, b}) in splits
        top1 = frozenset({a, c}) in splits
        top2 = frozenset({a, d}) in splits
        
        key = (a, b, c, d)
        if top0 and not top1 and not top2:
            quartet_support[key] = 0
        elif top1 and not top0 and not top2:
            quartet_support[key] = 1
        elif top2 and not top0 and not top1:
            quartet_support[key] = 2
        else:
            # Ambiguous or missing - skip
            pass
    
    return quartet_support


def _bootstrap_vote(seqs, labels, dmat_base, pmat_base, num_bootstraps, L, n):
    """
    Perform bootstrap resampling and vote on topologies.
    Returns the most common topology as a Newick string, or None.
    """
    global _start_time
    
    topology_counts = {}
    topology_newick = {}
    
    for b in range(num_bootstraps):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE_FRACTION * 270.0:
            break
        
        boot_indices = np.random.randint(0, L, size=L)
        boot_seqs = seqs[:, boot_indices]
        
        dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
        dmat_corrected = _bionj_correct(dmat_boot, pmat_boot, n, L)
        
        try:
            tree = _neighbor_joining_vectorized(dmat_corrected, labels)
        except Exception:
            continue
        
        splits = _extract_splits(tree, labels)
        key = _topology_key(splits)
        
        if key not in topology_counts:
            topology_counts[key] = 0
            topology_newick[key] = tree
        topology_counts[key] += 1
    
    if not topology_counts:
        return None
    
    best_key = max(topology_counts, key=topology_counts.get)
    return topology_newick[best_key]


def _nj_from_distances(dmat, labels):
    """Convenience wrapper for NJ from a distance matrix."""
    return _neighbor_joining_vectorized(dmat, labels)


def reconstruct(sequences):
    global _start_time, _instance_count
    
    if _start_time is None:
        _start_time = time.time()
    
    _instance_count += 1
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 3.0:
        return _star_tree(labels)
    
    # Encode sequences
    seqs = _encode_sequences(sequences, labels)
    L = seqs.shape[1]
    
    # 1. Compute Distance Matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # 2. Apply BIONJ correction
    dmat_corrected = _bionj_correct(dmat, pmat, n, L)
    
    # 3. Run NJ on corrected distances
    try:
        tree = _neighbor_joining_vectorized(dmat_corrected, labels)
    except Exception:
        try:
            tree = _neighbor_joining_vectorized(dmat, labels)
        except Exception:
            tree = _star_tree(labels)
    
    # 4. Bootstrap voting if time permits
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    # Estimate time per bootstrap
    # Distance matrix: O(n^2 * L), NJ: O(n^3)
    # For n=32, L=2000: ~1-2ms per bootstrap
    per_boot_time = 0.002 * (n / 16.0) ** 1.5
    boot_time_budget = remaining * 0.55
    if boot_time_budget > 0.03:
        num_bootstraps = int(boot_time_budget / per_boot_time)
        num_bootstraps = max(8, min(num_bootstraps, 150))
        
        boot_tree = _bootstrap_vote(seqs, labels, dmat, pmat, num_bootstraps, L, n)
        if boot_tree is not None:
            tree = boot_tree
    
    return tree
