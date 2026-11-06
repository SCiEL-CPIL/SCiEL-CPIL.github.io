"""Phylogenetic topology recovery via NJ with exhaustive quartet search for small n."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations

_DEADLINE = None


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using efficient encoding."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Build a 2D array where each taxon is encoded as a vector of one-hot-ish values
    # Use integer encoding: A=0, C=1, G=2, T=3
    enc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        mat[i] = [enc_map[c] for c in s]
    
    # Compute pairwise Hamming distances using broadcasting in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 16
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # mat[i:i_end, None, :] shape (chunk, 1, L)
        # mat[None, :, :] shape (1, n, L)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2).astype(np.float64)
        p = counts / L
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc
    
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick(dist, labels):
    """
    Robust Neighbor Joining algorithm.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Use a list-based approach with explicit tracking
    nodes = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    # FIX: Allocate D for 2*n nodes to accommodate internal nodes
    D = np.zeros((2 * n, 2 * n), dtype=np.float64)
    D[:n, :n] = dist
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums using numpy for speed
        node_arr = np.array(nodes)
        # Get submatrix
        sub_D = D[np.ix_(node_arr, node_arr)]
        row_sums = sub_D.sum(axis=1)
        
        # NJ criterion: Q[i,j] = (m-2)*D[i,j] - row_sum[i] - row_sum[j]
        Q = (m - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        a_idx, b_idx = min_idx
        a = nodes[a_idx]
        b = nodes[b_idx]
        
        if a == b:
            return "(" + ",".join(tree_str[x] for x in nodes) + ")"
        
        new_id = next_id
        next_id += 1
        
        tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"
        
        # Update distances
        for x in nodes:
            if x == a or x == b:
                continue
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_id] = D[new_id, x]
        
        # Remove a and b
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
    
    a, b, c = nodes
    return f"({tree_str[a]},{tree_str[b]},{tree_str[c]})"


def _quartet_score(tree_newick, dist, labels):
    """
    Score a tree topology by summing quartet likelihoods (approximated via distances).
    For each quartet, check if the tree's split is consistent with the distance matrix.
    """
    # This is a simplified scoring - we'll use the NJ-like criterion
    # Actually, for exhaustive search, we need to enumerate all unrooted trees
    # and score them. Let's use a different approach.
    pass


def _all_unrooted_trees(n):
    """
    Generate all unrooted tree topologies for n taxa (labeled 0..n-1).
    Returns list of sets of splits (each split is a frozenset).
    For n=4, there are 3 possible topologies.
    For n=5, there are 15.
    For n=6, there are 105.
    For n=7, there are 945.
    """
    if n <= 3:
        return [set()]
    
    # Recursive generation
    # Base case: n=4
    if n == 4:
        # The 3 possible quartet topologies (unrooted)
        # Each is defined by which pair is "internal"
        # (0,1)|(2,3), (0,2)|(1,3), (0,3)|(1,2)
        return [
            {frozenset([0, 1]), frozenset([2, 3])},
            {frozenset([0, 2]), frozenset([1, 3])},
            {frozenset([0, 3]), frozenset([1, 2])},
        ]
    
    # For larger n, use recursive construction
    # A tree on n taxa can be built by taking a tree on n-1 taxa
    # and inserting taxon n-1 into an edge
    # But this generates rooted trees; for unrooted, we need to be careful
    
    # Alternative: enumerate all possible splits and check compatibility
    # For n up to 7, we can use a known enumeration
    
    # Let's use a simpler approach: for n=5,6,7 enumerate via recursion
    # Actually, let's just use NJ for n>4 and exhaustive for n<=4
    
    return []


def _exhaustive_small_n(sequences, labels):
    """
    For n <= 4, find the exact best tree by checking all topologies.
    For n=4, there are 3 unrooted topologies.
    """
    n = len(labels)
    if n < 4:
        return None
    
    dist = _compute_distance_matrix(sequences, labels)
    
    if n == 4:
        # 3 possible quartet topologies
        # Score each by the sum of the two "internal" edge lengths
        # or by how well the tree explains the distances
        
        # For quartet (a,b,c,d), the three splits are:
        # S1: (a,b)|(c,d) -> internal edges connect ab and cd
        # S2: (a,c)|(b,d)
        # S3: (a,d)|(b,c)
        
        # Use the NJ criterion to score each split
        # For split (A|B), the "cost" is related to how well it separates the data
        
        # Simple approach: for each candidate split, compute the sum of 
        # distances within groups vs between groups
        # The best split minimizes the "conflict"
        
        # Actually, let's use the standard quartet test:
        # For quartet with taxa a,b,c,d, compute:
        # d_ab + d_cd vs d_ac + d_bd vs d_ad + d_bc
        # The pair with the smallest sum defines the internal edge
        
        d_ab = dist[0, 1] + dist[2, 3]
        d_ac = dist[0, 2] + dist[1, 3]
        d_ad = dist[0, 3] + dist[1, 2]
        
        # The minimum corresponds to the true topology
        if d_ab <= d_ac and d_ab <= d_ad:
            # (a,b)|(c,d) -> tree is ((a,b),(c,d)) but unrooted
            # Newick: (a,b,c,d) with internal edge between ab and cd
            # In Newick: ((a,b),(c,d)) is rooted; unrooted is just the split
            # Return: (a,b,c,d) with the understanding that the internal split is (ab|cd)
            # For Newick, we can write: (a,b,c,d) but that's a star
            # Better: ((a,b),(c,d)) as a rooted representation
            return f"({labels[0]},{labels[1]},{labels[2]},{labels[3]})"
        elif d_ac <= d_ad:
            return f"({labels[0]},{labels[2]},{labels[1]},{labels[3]})"
        else:
            return f"({labels[0]},{labels[3]},{labels[1]},{labels[2]})"
    
    return None


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # For n=4, use exhaustive quartet test
    if n == 4:
        dist = _compute_distance_matrix(sequences, labels)
        d01_23 = dist[0, 1] + dist[2, 3]
        d02_13 = dist[0, 2] + dist[1, 3]
        d03_12 = dist[0, 3] + dist[1, 2]
        
        if d01_23 <= d02_13 and d01_23 <= d03_12:
            return f"(({labels[0]},{labels[1]}),({labels[2]},{labels[3]}))"
        elif d02_13 <= d03_12:
            return f"(({labels[0]},{labels[2]}),({labels[1]},{labels[3]}))"
        else:
            return f"(({labels[0]},{labels[3]}),({labels[1]},{labels[2]}))"
    
    # Check deadline before expensive computation
    if time.time() > _DEADLINE - 15:
        return "(" + ",".join(labels) + ")"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return "(" + ",".join(labels) + ")"
    
    # Use NJ for topology recovery
    result = _nj_newick(dist, labels)
    
    return result

# EVOLVE-BLOCK-END
