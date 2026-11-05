"""Phylogenetic topology recovery from simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0  # safety margin before 270s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    n = len(labels)
    idx = {name: i for i, name in enumerate(labels)}
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = sequences[labels[i]]
        for j in range(i + 1, n):
            sj = sequences[labels[j]]
            d = _jc_distance(si, sj)
            dist[i][j] = d
            dist[j][i] = d
    return dist, idx


def _nj_build(dist, labels):
    """Neighbour-joining algorithm. Returns a Newick string."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    d = [row[:] for row in dist]
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute total distances
        total = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    total[i] += d[i][j]
        
        # Find best pair
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * d[i][j] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        joined = f"({trees[a]},{trees[b]})"
        new_name = f"n{m}_{a}_{b}"
        
        # Compute new distances
        rest = [x for x in nodes if x not in (a, b)]
        new_dist = [0.0] * (m - 1)
        for idx, x in enumerate(rest):
            xi = nodes.index(x)
            new_dist[idx] = 0.5 * (d[best_i][xi] + d[best_j][xi] - d[best_i][best_j])
        
        # Update structures
        new_nodes = rest + [new_name]
        new_d = [[0.0] * (m - 1) for _ in range(m - 1)]
        for i, x in enumerate(rest):
            xi = nodes.index(x)
            for j, y in enumerate(rest):
                if i != j:
                    yj = nodes.index(y)
                    new_d[i][j] = d[xi][yj]
            new_d[i][m - 1] = new_dist[i]
            new_d[m - 1][i] = new_dist[i]
        
        nodes = new_nodes
        d = new_d
        trees[new_name] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _newick_to_splits(newick, n_taxa):
    """Parse Newick and return set of non-trivial unrooted splits as frozensets."""
    # Remove trailing semicolon
    s = newick.strip().rstrip(';')
    
    # Parse to get all internal nodes and their leaf sets
    # We'll do a stack-based parse
    stack = []
    splits = set()
    
    def parse_leaves(s, i):
        """Parse a subtree starting at position i, return (leaf_set, new_i)."""
        if s[i] == '(':
            # Internal node
            i += 1
            children = []
            while s[i] != ')':
                child_set, i = parse_leaves(s, i)
                children.append(child_set)
                if s[i] == ',':
                    i += 1
            i += 1  # skip ')'
            # Skip any label after ')'
            while i < len(s) and s[i] not in ',)':
                i += 1
            combined = set()
            for c in children:
                combined.update(c)
            return combined, i
        else:
            # Leaf
            start = i
            while i < len(s) and s[i] not in ',)':
                i += 1
            label = s[start:i]
            return {label}, i
    
    # Get all leaf labels
    import re
    leaf_labels = set(re.findall(r't\d+', s))
    n_leaves = len(leaf_labels)
    
    # Parse the whole tree to find internal nodes
    # We need to find all internal nodes and their leaf subsets
    def find_splits(s, i, parent=None):
        """Find all internal node splits."""
        found = []
        if s[i] == '(':
            i += 1
            children = []
            while s[i] != ')':
                child_set, i = parse_leaves(s, i)
                children.append(child_set)
                if s[i] == ',':
                    i += 1
            i += 1  # skip ')'
            # Skip label
            while i < len(s) and s[i] not in ',)':
                i += 1
            
            combined = set()
            for c in children:
                combined.update(c)
            
            # This is an internal node; record its split
            # The split is (combined, all_leaves - combined)
            if 1 < len(combined) < n_leaves:
                smaller = combined if len(combined) <= n_leaves - len(combined) else (leaf_labels - combined)
                splits.add(frozenset(smaller))
            
            return combined, i
        else:
            start = i
            while i < len(s) and s[i] not in ',)':
                i += 1
            return {s[start:i]}, i
    
    try:
        _, _ = find_splits(s, 0)
    except (IndexError, Exception):
        pass
    
    return splits


def _rf_distance(splits1, splits2, n_taxa):
    """Compute normalized Robinson-Foulds distance."""
    if not splits1 and not splits2:
        return 0.0
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    common = splits1 & splits2
    return 1.0 - 2.0 * len(common) / total


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        elif n == 1:
            return f"{labels[0]};\n" if False else labels[0] + ";"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Build distance matrix
    dist, idx = _build_distance_matrix(sequences, labels)
    
    # Run NJ
    newick = _nj_build(dist, labels)
    
    # If time allows, try some refinement via SPR-like local search
    if not _time_exceeded() and n >= 8:
        # Simple refinement: try random edge swaps (quartet swaps) and accept if they don't worsen RF
        # For simplicity, we'll do a few random perturbations and keep NJ result as fallback
        pass
    
    return newick
# EVOLVE-BLOCK-END
