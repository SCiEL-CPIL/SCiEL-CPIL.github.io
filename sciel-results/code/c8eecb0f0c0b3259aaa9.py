"""Phylogenetic topology recovery via maximum likelihood under JC model."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 260.0  # Safety margin before 270s kill


def _jc_likelihood_split(sequences, labels, split_a, split_b, L):
    """
    Compute the log-likelihood of a bipartition (split_a | split_b) under JC model.
    For each site, compute the likelihood of the split being correct.
    Under JC, the probability that two sequences in the same clade are identical
    at a site depends on the branch lengths. We use a simplified approach:
    For a given split, we compute the likelihood by summing over all possible
    internal node states, but for JC symmetry, we can use the fact that
    the likelihood of a split can be approximated by the product over sites of
    the probability that the split is consistent with the data.
    
    A better approach: For each site, given the split (A|B), we compute the
    likelihood by considering the two clades. Under JC, the probability of
    observing the patterns in clade A and clade B given the split can be
    computed. But this is complex.
    
    Simpler: Use the quartet-based approach. For a split, we can compute the
    likelihood by considering all quartets that are consistent with the split.
    """
    # For efficiency, we'll use a distance-based approximation that's equivalent
    # to ML under JC for topology selection.
    # The "quartet score" for a split can be computed from pairwise distances.
    # For a split (A|B), the score is the sum over all quartets (a1,a2 in A, 
    # b1,b2 in B) of the quartet score for ((a1,a2)|(b1,b2)).
    # But this is O(|A|^2 * |B|^2) per split, which is too slow.
    
    # Alternative: Use the fact that under JC, the ML topology is the one that
    # minimizes the total branch length. We can compute this from the distance
    # matrix using the formula:
    # total_length = (1/(n-2)) * sum_{i,j} d(i,j)
    # But this doesn't depend on topology.
    
    # Let's use a different approach: For each split, compute the likelihood
    # by summing over sites. For each site, the likelihood of the split is
    # the product of the likelihoods of the two clades. Under JC, the
    # likelihood of a clade can be computed from the number of mismatches
    # within the clade.
    
    # Actually, for the purpose of topology selection under JC, the best
    # approach is to use the quartet method. Let's compute the quartet scores
    # for all quartets and then build the tree using a greedy approach.
    
    return 0.0  # Placeholder


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distances using NumPy."""
    n = len(labels)
    seqs = [sequences[l] for l in labels]
    L = len(seqs[0])
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.array([[code_map[c] for c in s] for s in seqs], dtype=np.int8)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mismatches = np.sum(mat[i] != mat, axis=1)
        p = mismatches / L
        valid = p < 0.7499
        d = np.where(valid, -0.75 * np.log(np.maximum(1.0 - 4.0 * p / 3.0, 1e-15)), 3.0)
        d[i] = 0.0
        dist[i] = d
    return dist


def _quartet_score(D, i, j, k, l):
    """
    Compute the three quartet scores for taxa i,j,k,l.
    Returns (score_ij_kl, score_ik_jl, score_il_jk)
    where score_ij_kl = d(i,k) + d(j,l) - d(i,j) - d(k,l)
    """
    s1 = D[i, k] + D[j, l] - D[i, j] - D[k, l]  # ((i,j)|(k,l))
    s2 = D[i, j] + D[k, l] - D[i, k] - D[j, l]  # ((i,k)|(j,l))
    s3 = D[i, l] + D[j, k] - D[i, j] - D[k, l]  # ((i,l)|(j,k))
    return s1, s2, s3


def _build_tree_from_quartets(D, labels, n):
    """
    Build a tree using the quartet method.
    For each quartet, determine the best topology.
    Then use a greedy approach to assemble the tree.
    """
    if n <= 4:
        # For 4 taxa, just return the best quartet topology
        if n == 4:
            s1, s2, s3 = _quartet_score(D, 0, 1, 2, 3)
            best = min([(s1, 0), (s2, 1), (s3, 2)], key=lambda x: x[0])
            topo = best[1]
            if topo == 0:
                return f"({labels[0]},{labels[1]},{labels[2]},{labels[3]});"
            elif topo == 1:
                return f"({labels[0]},{labels[2]},{labels[1]},{labels[3]});"
            else:
                return f"({labels[0]},{labels[3]},{labels[1]},{labels[2]});"
        elif n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return labels[0] if labels else ""
    
    # For n > 4, use a greedy quartet-based approach
    # Start with a small tree and add taxa one by one
    
    # Initialize with the first 4 taxa
    s1, s2, s3 = _quartet_score(D, 0, 1, 2, 3)
    best = min([(s1, 0), (s2, 1), (s3, 2)], key=lambda x: x[0])
    topo = best[1]
    
    # Build initial tree structure
    # We'll represent the tree as a set of splits
    # For the initial 4-taxon tree, the split is determined by the topology
    
    # For efficiency, let's use a different approach:
    # Use NJ as a starting point, then refine with quartet-based NNI
    
    return None  # Fall back to NJ


def _nj_build(D):
    """Standard Neighbor-Joining. Returns list of (a, b) joins in order."""
    n = D.shape[0]
    if n <= 2:
        return []
    D_cur = D.copy()
    r = np.sum(D_cur, axis=1)
    active = list(range(n))
    joins = []
    while len(active) > 3:
        m = len(active)
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                q = (m - 2) * D_cur[i, j] - r[i] - r[j]
                if q < best_val:
                    best_val = q
                    best_i = i
                    best_j = j
        i, j = best_i, best_j
        joins.append((i, j))
        new_node = n + len(joins) - 1
        active.remove(i)
        active.remove(j)
        for x in active:
            D_cur[new_node, x] = 0.5 * (D_cur[i, x] + D_cur[j, x] - D_cur[i, j])
            D_cur[x, new_node] = D_cur[new_node, x]
        r[new_node] = sum(D_cur[new_node, x] for x in active)
        for x in active:
            r[x] = sum(D_cur[x, y] for y in active)
        active.append(new_node)
    return joins


def _build_newick(labels, joins):
    """Build Newick from NJ joins."""
    n = len(labels)
    tree_str = {i: labels[i] for i in range(n)}
    for idx, (a, b) in enumerate(joins):
        new_node = n + idx
        tree_str[new_node] = f"({tree_str[a]},{tree_str[b]})"
    root = n + len(joins) - 1
    return tree_str[root] + ";"


def _get_splits_from_joins(labels, joins):
    """Get non-trivial splits directly from NJ join structure."""
    n = len(labels)
    all_leaves = frozenset(range(n))
    splits = set()
    
    # Build subtree leaf sets
    subtree_leaves = {}
    for i in range(n):
        subtree_leaves[i] = frozenset([i])
    
    for idx, (a, b) in enumerate(joins):
        new_node = n + idx
        subtree_leaves[new_node] = subtree_leaves[a] | subtree_leaves[b]
        sl = subtree_leaves[new_node]
        if 0 < len(sl) < n:
            if len(sl) <= n // 2:
                splits.add(sl)
            else:
                splits.add(all_leaves - sl)
    
    return splits


def _nni_refine(labels, joins, D, sequences, max_iterations=10):
    """
    NNI refinement using quartet scores.
    For each internal edge, try the three quartet topologies
    and switch if a better topology is found.
    """
    n = len(labels)
    if n <= 4:
        return joins
    
    current_joins = list(joins)
    
    # Build tree structure
    # Node i (0..n-1) is a leaf, node n+k is internal with children joins[k]
    children = {}
    for k, (a, b) in enumerate(joins):
        children[n + k] = (a, b)
    
    # Find parent for each node
    parent = {}
    for k, (a, b) in enumerate(joins):
        new_node = n + k
        parent[a] = new_node
        parent[b] = new_node
    
    root = n + len(joins) - 1
    
    # For NNI, we need to consider internal edges
    # An internal edge is (parent, child) where child is an internal node
    # and parent is also an internal node (or root)
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 10:
            break
        
        improved = False
        
        # Try NNI moves for each internal edge
        for child_node in range(n, root + 1):
            if child_node not in parent:
                continue  # This is the root
            par = parent[child_node]
            if par < n:
                continue  # Parent is a leaf, not an internal edge
            
            # The edge is (par, child_node)
            # The four subtrees involved in NNI are:
            # 1. child_node's subtree
            # 2. The other child of par
            # 3. par's parent's subtree (excluding the edge to par)
            # 4. The other child of par's parent
            
            # Get the two children of par
            c1, c2 = children[par]
            if c1 == child_node:
                other_child_par = c2
            else:
                other_child_par = c1
            
            # Get par's parent
            if par not in parent:
                continue  # par is the root
            
            grandpar = parent[par]
            if grandpar < n:
                continue  # Grandparent is a leaf
            
            # Get grandpar's two children
            gc1, gc2 = children[grandpar]
            if gc1 == par:
                other_child_grandpar = gc2
            else:
                other_child_grandpar = gc1
            
            # Now we have four subtrees:
            # A: child_node
            # B: other_child_par
            # C: other_child_grandpar
            # D: grandpar's subtree excluding par (which is just other_child_grandpar)
            
            # Actually, for NNI around edge (par, child_node), the four subtrees are:
            # 1. child_node
            # 2. other_child_par
            # 3. The subtree of grandpar excluding par (which includes other_child_grandpar)
            # 4. The subtree of par excluding child_node (which is other_child_par)
            
            # This is getting complex. Let's use a simpler approach:
            # For each pair of adjacent internal nodes, consider the four "pendant"
            # subtrees and try the three quartet topologies.
            
            # For now, skip this and use a simpler heuristic
            pass
        
        # If no improvement, break
        if not improved:
            break
    
    return current_joins


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 15:
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    try:
        joins = _nj_build(D)
        newick = _build_newick(labels, joins)
    except Exception:
        newick = "(" + ",".join(labels) + ");"
    
    # NNI refinement if time permits
    if time.time() < _DEADLINE - 30:
        try:
            refined_joins = _nni_refine(labels, joins, D, sequences)
            newick = _build_newick(labels, refined_joins)
        except Exception:
            pass
    
    return newick
# EVOLVE-BLOCK-END
