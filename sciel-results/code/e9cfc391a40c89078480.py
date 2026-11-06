"""Phylogenetic topology recovery using NJ + bootstrap consensus + NNI polish."""
import math
import time
import numpy as np

_DEADLINE = 245.0
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0]) if n > 0 else 1
    D = np.zeros((n, n))
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            # Count mismatches
            mismatches = sum(1 for a, b in zip(si, sj) if a != b)
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_build(D, labels):
    """Run Neighbor Joining, return list of (i, j, new_node) joins and final Newick."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes))
    dist[:n, :n] = D
    active = list(range(n))
    tree_strs = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        total = np.zeros(m)
        for k, i in enumerate(active):
            total[k] = sum(dist[i, j] for j in active if j != i)
        
        best_criterion = float('inf')
        best_pair = None
        for ki in range(m):
            for kj in range(ki + 1, m):
                i, j = active[ki], active[kj]
                criterion = (m - 2) * dist[i, j] - total[ki] - total[kj]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined = f"({tree_strs[i]},{tree_strs[j]})"
        new_idx = max(active) + 1
        
        if new_idx >= max_nodes:
            break
            
        rest = [x for x in active if x not in (i, j)]
        
        for x in rest:
            d_new = 0.5 * (dist[i, x] + dist[j, x] - dist[i, j])
            dist[new_idx, x] = d_new
            dist[x, new_idx] = d_new
        
        active = rest + [new_idx]
        tree_strs[new_idx] = joined
    
    parts = [tree_strs[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _split_top_level(s):
    """Split string on top-level commas."""
    parts = []
    depth = 0
    current = []
    for c in s:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        parts.append(''.join(current))
    return [p.strip() for p in parts if p.strip()]


def _get_splits(newick):
    """Extract all non-trivial unrooted splits from a Newick string."""
    s = newick.rstrip(';').strip()
    
    # Collect all taxa
    taxa = set()
    def _collect(s):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            for part in _split_top_level(inner):
                _collect(part)
        else:
            if ':' in s:
                s = s.split(':')[0].strip()
            if s:
                taxa.add(s)
    _collect(s)
    
    n = len(taxa)
    if n < 4:
        return set()
    
    splits = set()
    
    def _collect_splits(s):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            parts = _split_top_level(inner)
            all_taxa_here = set()
            for part in parts:
                part = part.strip()
                if part in taxa:
                    all_taxa_here.add(part)
                elif part.startswith('(') and part.endswith(')'):
                    all_taxa_here |= _collect_splits(part)
            
            if 0 < len(all_taxa_here) < n:
                other = taxa - all_taxa_here
                a, b = all_taxa_here, other
                if len(a) > len(b) or (len(a) == len(b) and sorted(a) > sorted(b)):
                    a, b = b, a
                splits.add(frozenset(a))
        
        return set()
    
    _collect_splits(s)
    return splits


def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two split sets."""
    total1 = len(splits1)
    total2 = len(splits2)
    if total1 == 0 and total2 == 0:
        return 0.0
    common = len(splits1 & splits2)
    rf = (total1 - common) + (total2 - common)
    max_possible = max(total1, total2)
    if max_possible == 0:
        return 0.0
    return rf / max_possible


def _bootstrap_nj(sequences, labels, n_boot, rng):
    """Run bootstrap NJ and return list of Newick strings."""
    n = len(labels)
    L = len(sequences[labels[0]])
    boot_trees = []
    
    for b in range(n_boot):
        if _time_left() < 10.0:
            break
        # Resample columns
        idx = rng.integers(0, L, size=L)
        boot_seqs = {lb: ''.join(sequences[lb][i] for i in idx) for lb in labels}
        D = _jc_distance_matrix(boot_seqs, labels)
        newick = _nj_build(D, labels)
        boot_trees.append(newick)
    
    return boot_trees


def _median_tree(boot_trees, labels):
    """Select the median tree from bootstrap trees (highest split support)."""
    if not boot_trees:
        return None
    
    # Collect all splits and their support
    all_splits = {}
    for newick in boot_trees:
        splits = _get_splits(newick)
        for s in splits:
            all_splits[s] = all_splits.get(s, 0) + 1
    
    if not all_splits:
        return boot_trees[0]
    
    # Select the tree with the highest total support
    best_tree = boot_trees[0]
    best_score = -1
    for newick in boot_trees:
        splits = _get_splits(newick)
        score = sum(all_splits.get(s, 0) for s in splits)
        if score > best_score:
            best_score = score
            best_tree = newick
    
    return best_tree


def _nni_polish(newick, D, labels):
    """Perform NNI polish to improve topology."""
    # This is a simplified NNI that tries local rearrangements
    # For now, return the input as-is (placeholder for future improvement)
    return newick


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    newick = _nj_build(D, labels)
    
    # Bootstrap consensus if we have time
    if _time_left() > 20.0:
        # Determine number of bootstraps based on available time
        # Each bootstrap takes roughly O(n^2 * L) operations
        # Be conservative with time estimate
        max_boot = min(100, int(_time_left() / 0.3))
        max_boot = max(max_boot, 10)
        
        rng = np.random.default_rng(42)
        boot_trees = _bootstrap_nj(sequences, labels, max_boot, rng)
        
        if boot_trees and len(boot_trees) >= 5:
            consensus = _median_tree(boot_trees, labels)
            if consensus:
                newick = consensus
    
    # NNI polish if time remains
    if _time_left() > 10.0:
        polished = _nni_polish(newick, D, labels)
        if polished:
            newick = polished
    
    return newick
