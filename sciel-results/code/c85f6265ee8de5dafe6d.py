"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''
Phylogenetic topology recovery using:
1. Fast NJ (Bionject) for initial tree
2. Distance-based refinement via split evaluation
3. Time-aware deadline guard
'''
import time
import math
import numpy as np
from collections import defaultdict

# Deadline guard: 270s wall-clock, keep 10s margin
_DEADLINE = 260.0
_start_time = time.time()

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix using numpy for speed."""
    n = len(labels)
    # Convert sequences to numpy arrays of integers
    seq_list = [sequences[l] for l in labels]
    L = len(seq_list[0])
    
    # Encode A=0, C=1, G=2, T=3
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            mat[i, j] = enc[c]
    
    # Compute pairwise differences using broadcasting
    # This can be memory-intensive for large n*L, so do it in chunks
    dist = np.zeros((n, n))
    for i in range(n):
        # Compute differences for row i against all others
        diffs = (mat[i:i+1, :] != mat[:, :]).sum(axis=1)
        p = diffs / L
        # JC correction
        p = np.clip(p, 0, 0.7499)
        dist[i, :] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return dist

def _bionject(dist, labels):
    """
    Bionject (fast NJ) algorithm.
    dist: (n, n) numpy array
    labels: list of taxon labels
    Returns: Newick string
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Use a list to track current nodes
    nodes = list(range(n))  # indices into original labels
    trees = {i: labels[i] for i in range(n)}
    # dist is already computed
    D = dist.copy()
    
    while len(nodes) > 3:
        if _check_deadline():
            break
        n_cur = len(nodes)
        # Compute row sums
        row_sums = np.zeros(n_cur)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i != j:
                    row_sums[i] += D[ni, nj]
        
        # Bionject criterion: Q(i,j) = (n-2)*d(i,j) - r(i) - r(j)
        best_q = float('inf')
        best_i = best_j = -1
        for i in range(n_cur):
            for j in range(i+1, n_cur):
                ni, nj = nodes[i], nodes[j]
                q = (n_cur - 2) * D[ni, nj] - row_sums[i] - row_sums[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
            
        ni, nj = nodes[best_i], nodes[best_j]
        new_idx = max(range(n)) + 1
        new_label = f"n{new_idx}"
        
        # New tree string
        trees[new_idx] = f"({trees[ni]},{trees[nj]})"
        
        # Compute new distances
        rest = [x for x in nodes if x not in (ni, nj)]
        for x in rest:
            d_new = 0.5 * (D[ni, x] + D[nj, x] - D[ni, nj])
            D[new_idx, x] = D[x, new_idx] = d_new
        
        # Remove old nodes, add new
        nodes = rest + [new_idx]
    
    # Build final tree from remaining nodes
    if len(nodes) == 3:
        a, b, c = nodes
        # For 3 taxa, just combine them
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(nodes) == 2:
        return f"({trees[nodes[0]]},{trees[nodes[1]]});"
    elif len(nodes) == 1:
        return trees[nodes[0]] + ";"
    else:
        # Should not happen, but fallback
        return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns: (children_dict, parent_dict, root)
    children_dict: node -> list of children
    parent_dict: node -> parent
    root: root node
    """
    newick = newick.strip().rstrip(';')
    children = defaultdict(list)
    parent = {}
    stack = []
    i = 0
    node_counter = 0
    node_names = {}
    
    while i < len(newick):
        c = newick[i]
        if c == '(':
            # Create new internal node
            node_counter += 1
            name = f"node_{node_counter}"
            node_names[name] = name
            if stack:
                parent[name] = stack[-1]
                children[stack[-1]].append(name)
            stack.append(name)
        elif c == ')':
            # Close current node
            if stack:
                stack.pop()
        elif c == ',':
            # Sibling separator
            pass
        else:
            # Taxon name
            j = i
            while j < len(newick) and newick[j] not in (',', ')', '(', ';'):
                j += 1
            name = newick[i:j]
            if stack:
                parent[name] = stack[-1]
                children[stack[-1]].append(name)
            i = j - 1
        i += 1
    
    root = stack[0] if stack else None
    if root is None and children:
        # Find root
        for n in children:
            if n not in parent:
                root = n
                break
    
    return children, parent, root

def _get_splits(children, parent, root, taxa):
    """
    Get all non-trivial splits from a tree.
    A split is a partition of taxa into two non-empty sets.
    We represent each split as a frozenset of the smaller side.
    """
    splits = set()
    
    def _get_subtree_taxa(node):
        """Get all taxa in the subtree rooted at node."""
        if node in taxa:
            return {node}
        result = set()
        for child in children.get(node, []):
            result |= _get_subtree_taxa(child)
        return result
    
    # For each internal node, get the split it defines
    for node in list(children.keys()):
        if node in taxa:
            continue
        subtree = _get_subtree_taxa(node)
        complement = set(taxa) - subtree
        if 0 < len(subtree) < len(taxa):
            # Normalize: use the smaller side
            if len(subtree) <= len(complement):
                splits.add(frozenset(subtree))
            else:
                splits.add(frozenset(complement))
    
    return splits

def _robinson_foulds(splits1, splits2):
    """Compute normalized RF distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    common = splits1 & splits2
    total = len(splits1 | splits2)
    if total == 0:
        return 0.0
    return (total - 2 * len(common)) / total

def _refine_tree(newick, sequences, labels, dist):
    """
    Try to refine the tree by evaluating splits.
    This is a simplified approach - we'll try a few random perturbations.
    """
    if _check_deadline():
        return newick
    
    children, parent, root = _parse_newick(newick)
    taxa = set(labels)
    current_splits = _get_splits(children, parent, root, taxa)
    current_rf = len(current_splits)  # Just track count for now
    
    return newick  # Simplified: return as-is for now

def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree from DNA sequences.
    Returns Newick string.
    """
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # Run Bionject (fast NJ)
    tree = _bionject(dist, labels)
    
    # Check deadline before refinement
    if _check_deadline():
        return tree
    
    # Basic refinement: try to improve by re-evaluating
    # For now, just return the NJ tree
    return tree
# EVOLVE-BLOCK-END
