"""Phylogenetic topology recovery: spectral embedding + hierarchical clustering with NJ fallback."""
import time
import math
import numpy as np
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = 240.0  # safety margin before 270s kill

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix using numpy for speed."""
    n = len(labels)
    seq_list = [sequences[l] for l in labels]
    L = len(seq_list[0])
    
    # Encode A=0, C=1, G=2, T=3
    enc = np.array([{'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] for c in 'ACGT'], dtype=np.int8)
    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            mat[i, j] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[c]
    
    # Compute pairwise differences using broadcasting
    dist = np.zeros((n, n))
    for i in range(n):
        diffs = (mat[i:i+1, :] != mat[:, :]).sum(axis=1)
        p = diffs / L
        p = np.clip(p, 0, 0.7499)
        dist[i, :] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Ensure symmetry
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist

def _spectral_embed(dist, n_components=None):
    """
    Embed distance matrix via double centering and eigendecomposition.
    Returns coordinates in R^k for k components.
    """
    n = len(dist)
    if n_components is None:
        n_components = min(n - 2, 10)
    
    # Double centering: B = -0.5 * J * D^2 * J
    # where J = I - (1/n) * 11^T
    D2 = dist ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top k components with positive eigenvalues
    k = min(n_components, max(1, int(np.sum(eigenvalues > 1e-10))))
    if k < 1:
        k = 1
    
    coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0))
    return coords

def _ward_linkage(coords, labels):
    """
    Perform Ward's linkage hierarchical clustering.
    Returns a list of (cluster_a, cluster_b, distance) merges in order.
    """
    n = len(labels)
    if n <= 2:
        return []
    
    # Initialize: each point is its own cluster
    clusters = [{'ids': [i], 'size': 1, 'centroid': coords[i:i+1]} for i in range(n)]
    active = list(range(n))
    merges = []
    
    while len(active) > 1:
        if _check_deadline():
            break
        
        # Find the pair of clusters with minimum Ward merge cost
        min_cost = np.inf
        min_pair = (-1, -1)
        
        for i_idx in range(len(active)):
            for j_idx in range(i_idx + 1, len(active)):
                ci = active[i_idx]
                cj = active[j_idx]
                
                # Ward's method: cost = (n_i * n_j) / (n_i + n_j) * ||c_i - c_j||^2
                ni = clusters[ci]['size']
                nj = clusters[cj]['size']
                cost = (ni * nj) / (ni + nj) * np.sum((clusters[ci]['centroid'] - clusters[cj]['centroid']) ** 2)
                
                if cost < min_cost:
                    min_cost = cost
                    min_pair = (ci, cj)
        
        if min_pair[0] == -1:
            break
        
        ci, cj = min_pair
        ni = clusters[ci]['size']
        nj = clusters[cj]['size']
        
        # Merge
        new_ids = clusters[ci]['ids'] + clusters[cj]['ids']
        new_centroid = (ni * clusters[ci]['centroid'] + nj * clusters[cj]['centroid']) / (ni + nj)
        
        merges.append((ci, cj, min_cost))
        
        # Create new cluster
        new_cluster = {'ids': new_ids, 'size': ni + nj, 'centroid': new_centroid}
        new_idx = len(clusters)
        clusters.append(new_cluster)
        
        # Remove old clusters from active
        active.remove(ci)
        active.remove(cj)
        active.append(new_idx)
    
    return merges, clusters

def _build_newick_from_merges(merges, clusters, labels):
    """
    Build Newick string from Ward linkage merges.
    """
    if not merges:
        return "(" + ",".join(labels) + ");"
    
    # Build tree structure
    # Each merge creates a new internal node
    # node_map: cluster_index -> newick_string
    node_map = {}
    for i, label in enumerate(labels):
        node_map[i] = label
    
    # Process merges in reverse order (last merge is the root)
    for ci, cj, cost in reversed(merges):
        new_idx = len(node_map)
        node_map[new_idx] = f"({node_map[ci]},{node_map[cj]})"
    
    # The root is the last cluster
    root_idx = len(clusters) - 1
    return node_map[root_idx] + ";"

def _nj_newick(dist, labels):
    """Fast NJ (Bionject) algorithm."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Pre-allocate distance matrix
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active_nodes = list(range(n))
    node_newick = {i: labels[i] for i in range(n)}
    next_idx = n
    
    while len(active_nodes) > 3:
        if _check_deadline():
            break
            
        n_cur = len(active_nodes)
        sub_D = D[np.ix_(active_nodes, active_nodes)]
        
        row_sums = sub_D.sum(axis=1)
        Q = (n_cur - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_local, j_local = min_idx
        
        if i_local > j_local:
            i_local, j_local = j_local, i_local
            
        ni = active_nodes[i_local]
        nj = active_nodes[j_local]
        
        new_idx = next_idx
        next_idx += 1
        
        node_newick[new_idx] = f"({node_newick[ni]},{node_newick[nj]})"
        
        rest = [x for k, x in enumerate(active_nodes) if k != i_local and k != j_local]
        if rest:
            d_ni_nj = D[ni, nj]
            d_ni_rest = D[ni, rest]
            d_nj_rest = D[nj, rest]
            new_dists = 0.5 * (d_ni_rest + d_nj_rest - d_ni_nj)
            
            D[new_idx, rest] = new_dists
            D[rest, new_idx] = new_dists
        
        active_nodes = rest + [new_idx]
    
    if len(active_nodes) == 3:
        a, b, c = active_nodes
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]});"
    elif len(active_nodes) == 2:
        return f"({node_newick[active_nodes[0]]},{node_newick[active_nodes[1]]});"
    elif len(active_nodes) == 1:
        return node_newick[active_nodes[0]] + ";"
    else:
        return "(" + ",".join(node_newick[x] for x in active_nodes) + ";"

def _parse_newick(newick):
    """Parse Newick string into tree structure."""
    newick = newick.strip().rstrip(';')
    children = defaultdict(list)
    parent = {}
    stack = []
    i = 0
    node_counter = 0
    
    while i < len(newick):
        c = newick[i]
        if c == '(':
            node_counter += 1
            name = f"node_{node_counter}"
            if stack:
                parent[name] = stack[-1]
                children[stack[-1]].append(name)
            stack.append(name)
        elif c == ')':
            if stack:
                stack.pop()
        elif c == ',':
            pass
        else:
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
        for n in children:
            if n not in parent:
                root = n
                break
    
    return children, parent, root

def _get_splits(children, parent, root, taxa):
    """Get all non-trivial splits from a tree."""
    splits = set()
    taxa_set = set(taxa)
    
    def _get_subtree_taxa(node):
        if node in taxa_set:
            return {node}
        result = set()
        for child in children.get(node, []):
            result |= _get_subtree_taxa(child)
        return result
    
    for node in list(children.keys()):
        if node in taxa_set:
            continue
        subtree = _get_subtree_taxa(node)
        complement = taxa_set - subtree
        if 0 < len(subtree) < len(taxa_set):
            if len(subtree) <= len(complement):
                splits.add(frozenset(subtree))
            else:
                splits.add(frozenset(complement))
    
    return splits

def _total_tree_length(newick, dist, labels):
    """Calculate total branch length of tree using distance matrix."""
    children, parent, root = _parse_newick(newick)
    taxa = set(labels)
    
    # Compute branch lengths using least squares
    # For each internal node, solve for branch lengths
    # This is complex, so use a simpler heuristic
    
    # Alternative: sum of distances between leaves in same clade / 2
    # This is a proxy for total tree length
    
    total = 0.0
    for node in children:
        if node in taxa:
            continue
        # Get taxa in subtree
        subtree_taxa = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n in taxa:
                subtree_taxa.add(n)
            else:
                stack.extend(children.get(n, []))
        
        if len(subtree_taxa) >= 2:
            # Sum pairwise distances in clade
            clade_dist = 0.0
            taxa_list = list(subtree_taxa)
            for i in range(len(taxa_list)):
                for j in range(i+1, len(taxa_list)):
                    idx_i = labels.index(taxa_list[i])
                    idx_j = labels.index(taxa_list[j])
                    clade_dist += dist[idx_i, idx_j]
            total += clade_dist / 2.0
    
    return total

def reconstruct(sequences):
    """Reconstruct phylogenetic tree from DNA sequences."""
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # Try spectral embedding + Ward linkage
    best_tree = None
    best_score = np.inf
    
    # Method 1: Spectral embedding + Ward linkage
    try:
        coords = _spectral_embed(dist)
        if coords is not None and len(coords) == n:
            merges, clusters = _ward_linkage(coords, labels)
            if merges:
                tree = _build_newick_from_merges(merges, clusters, labels)
                # Validate: check all taxa present
                taxa_in_tree = set()
                for l in labels:
                    if l in tree:
                        taxa_in_tree.add(l)
                if len(taxa_in_tree) == n:
                    # Score it
                    children, parent, root = _parse_newick(tree)
                    splits = _get_splits(children, parent, root, labels)
                    score = _total_tree_length(tree, dist, labels)
                    if score < best_score:
                        best_score = score
                        best_tree = tree
    except Exception:
        pass
    
    # Method 2: NJ fallback
    if _check_deadline():
        return best_tree if best_tree else _nj_newick(dist, labels)
    
    nj_tree = _nj_newick(dist, labels)
    if nj_tree:
        children, parent, root = _parse_newick(nj_tree)
        splits = _get_splits(children, parent, root, labels)
        score = _total_tree_length(nj_tree, dist, labels)
        if score < best_score:
            best_score = score
            best_tree = nj_tree
    
    # Return best tree found
    return best_tree if best_tree else _nj_newick(dist, labels)
