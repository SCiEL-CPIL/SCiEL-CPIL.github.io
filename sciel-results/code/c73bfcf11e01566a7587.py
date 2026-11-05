import numpy as np
import time
import math
from itertools import combinations

def _jc_distance_matrix(sequences):
    """Compute Jukes-Cantor distance matrix from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[char_to_int[c] for c in sequences[label]] for label in labels], dtype=np.uint8)
    L = seqs.shape[1]
    
    # Compute pairwise mismatch counts using broadcasting
    # For memory efficiency with larger n, we can do this in chunks
    # But for n<=32, L<=2000, it's fine: 32*32*2000 = 2M elements
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2).astype(np.float64)
    p = mismatches / L
    
    # Jukes-Cantor correction
    d = np.zeros((n, n), dtype=np.float64)
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    """Fast Neighbor Joining algorithm."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Use a distance matrix that we update in-place
    D = dist.copy()
    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Newick strings for each node
    trees = {i: labels[i] for i in range(n)}
    # Counter for new node IDs
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r_i = sum of distances from i to all other active nodes
        r = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(active):
            for jdx, j in enumerate(active):
                if jdx != idx:
                    r[idx] += D[i, j]
        
        # Find min Q_ij
        best_q = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = active[idx_i]
                j = active[idx_j]
                q = (m - 2) * D[i, j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i_idx = idx_i
                    best_j_idx = idx_j
        
        i = active[best_i_idx]
        j = active[best_j_idx]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick string for new node
        trees[new_id] = f"({trees[i]},{trees[j]})"
        
        # Update distances from new node to other active nodes
        for x in active:
            if x != i and x != j:
                d_new_x = 0.5 * (D[i, x] + D[j, x] - D[i, j])
                D[new_id, x] = d_new_x
                D[x, new_id] = d_new_x
        
        # Remove i and j from active, add new_id
        active = [x for x in active if x not in (i, j)]
        active.append(new_id)
        
        # Ensure D has enough space for new_id
        # Since we're using a fixed-size D, we need to be careful
        # Actually, D is n x n initially, but we're adding new nodes
        # We need to expand D or use a dictionary
        # For simplicity, let's use a dictionary-based approach for large n
        # But for n<=32, we can just pre-allocate a larger matrix
        
    # Final 3 nodes
    a, b, c = active
    return f"({trees[a]},({trees[b]},{trees[c]}));"

def _nj_with_expansion(dist, labels):
    """NJ with pre-allocated expanded distance matrix."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Maximum number of nodes we'll have is 2*n - 2
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r vector
        r = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(active):
            for jdx, j in enumerate(active):
                if jdx != idx:
                    r[idx] += D[i, j]
        
        # Find min Q
        best_q = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = active[idx_i]
                j = active[idx_j]
                q = (m - 2) * D[i, j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i_idx = idx_i
                    best_j_idx = idx_j
        
        i = active[best_i_idx]
        j = active[best_j_idx]
        
        new_id = next_id
        next_id += 1
        
        trees[new_id] = f"({trees[i]},{trees[j]})"
        
        for x in active:
            if x != i and x != j:
                d_new_x = 0.5 * (D[i, x] + D[j, x] - D[i, j])
                D[new_id, x] = d_new_x
                D[x, new_id] = d_new_x
        
        active = [x for x in active if x not in (i, j)]
        active.append(new_id)
    
    a, b, c = active
    return f"({trees[a]},({trees[b]},{trees[c]}));"

def _quartet_census(sequences, labels, max_quartets=None):
    """
    Compute quartet support for all 4-taxon subsets.
    Returns a dict mapping frozenset of 4 taxa to (support_012, support_013, support_023)
    where these correspond to the three possible unrooted quartet topologies.
    """
    n = len(labels)
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[char_to_int[c] for c in sequences[label]] for label in labels], dtype=np.uint8)
    L = seqs.shape[1]
    
    # Precompute pairwise mismatch matrices for each site
    # Actually, let's compute per-site consensus or use a different approach
    
    # For each quartet (a,b,c,d), count sites where:
    # Topology 1: (a,b)|(c,d) - a,b agree and c,d agree, but the groups differ
    # Topology 2: (a,c)|(b,d)
    # Topology 3: (a,d)|(b,c)
    
    # A site supports (a,b)|(c,d) if seq[a]==seq[b] and seq[c]==seq[d] and seq[a]!=seq[c]
    # But this is not quite right for JC model.
    
    # Better: use the "maximum likelihood" quartet score.
    # For JC, the log-likelihood ratio can be computed from the number of
    # sites with each pattern.
    
    # For a quartet with leaves a,b,c,d, there are 4^4 = 256 possible site patterns.
    # But due to symmetry, we can group them.
    
    # Let's use a simpler approach: count the number of sites that are
    # "compatible" with each topology.
    
    # For topology (a,b)|(c,d):
    # The site pattern is "compatible" if the two characters in {a,b} are the same
    # and the two characters in {c,d} are the same, and the two groups differ.
    # But this is too restrictive.
    
    # Actually, for the quartet method, we should compute the log-likelihood
    # under each of the three topologies for each site, and sum.
    
    # For JC model, the log-likelihood for a single site under a given topology
    # can be computed using the Felsenstein pruning algorithm.
    
    # But that's complex. Let's use a simpler heuristic:
    # Count the number of sites where the pattern is "informative" for each topology.
    
    # An informative site for (a,b)|(c,d) is one where:
    # seq[a] == seq[b] and seq[c] == seq[d] and seq[a] != seq[c]
    
    # This is the "parsimony-informative" site count.
    
    # Let's compute this for all quartets.
    
    quartet_support = {}
    quartets = list(combinations(range(n), 4))
    
    if max_quartets is not None and len(quartets) > max_quartets:
        # Sample a subset of quartets
        np.random.seed(42)
        indices = np.random.choice(len(quartets), max_quartets, replace=False)
        quartets = [quartets[i] for i in indices]
    
    for a, b, c, d in quartets:
        sa, sb, sc, sd = seqs[a], seqs[b], seqs[c], seqs[d]
        
        # Count parsimony-informative sites for each topology
        # (a,b)|(c,d): sa==sb, sc==sd, sa!=sc
        # (a,c)|(b,d): sa==sc, sb==sd, sa!=sb
        # (a,d)|(b,c): sa==sd, sb==sc, sa!=sb
        
        # Vectorized computation
        ab_cd = np.sum((sa == sb) & (sc == sd) & (sa != sc))
        ac_bd = np.sum((sa == sc) & (sb == sd) & (sa != sb))
        ad_bc = np.sum((sa == sd) & (sb == sc) & (sa != sb))
        
        key = frozenset((labels[a], labels[b], labels[c], labels[d]))
        quartet_support[key] = (ab_cd, ac_bd, ad_bc)
    
    return quartet_support

def _assemble_tree_from_quartets(quartet_support, labels, n):
    """
    Assemble a tree from quartet support using a greedy algorithm.
    This is a simplified version of the quartet method.
    """
    # For now, fall back to NJ if quartet assembly is too complex
    # This is a placeholder
    return None

def reconstruct(sequences):
    start_time = time.time()
    deadline = 265.0  # Safety margin before 270s kill
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    # Try NJ first (fast and reliable)
    tree_newick = _nj_with_expansion(d, labels)
    
    # Check time
    if time.time() - start_time > deadline:
        return tree_newick
    
    # If we have time, try quartet-based refinement
    # But for now, just return NJ result
    # The quartet method is complex to implement correctly within the time budget
    
    return tree_newick
