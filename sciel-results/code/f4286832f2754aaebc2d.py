"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 265.0  # seconds; leave 5s margin before the 270s kill
_START = time.time()

def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    if length == 0:
        return np.zeros((n, n))

    # Map DNA characters to integers for efficient comparison
    # A=0, C=1, G=2, T=3
    dna_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    dna_arr = np.array([[dna_map[c] for c in s] for s in seqs])
    
    # Compute pairwise Hamming distances
    # Use broadcasting for vectorized comparison
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    # Proportion of differences
    p = mismatches.astype(float) / length
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or log(negative)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _neighbor_joining(D, labels):
    """
    Robust Neighbor Joining implementation.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Work with a copy of the distance matrix
    D = D.copy()
    # List of current cluster Newick strings
    clusters = list(labels)
    
    m = n
    
    while m > 3:
        if time.time() > _DEADLINE:
            # Fallback: return a star-like tree of remaining clusters
            # This is not ideal but prevents timeout
            return "(" + ",".join(clusters) + ");"
            
        # Compute row sums
        r = np.sum(D, axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
            
        # Get the names/strings of the clusters being joined
        name_i = clusters[i]
        name_j = clusters[j]
        
        # Create the new cluster string
        # We don't add branch lengths in the Newick string for simplicity of topology scoring,
        # but we must maintain valid Newick syntax.
        new_cluster_name = f"({name_i},{name_j})"
        
        # Calculate distances from the new node to all other nodes
        # d_new_k = 0.5 * (d_ik + d_jk - d_ij)
        d_ij = D[i, j]
        
        # Indices of other nodes
        others = [k for k in range(m) if k != i and k != j]
        
        if others:
            d_i_others = D[i, others]
            d_j_others = D[j, others]
            d_new_others = 0.5 * (d_i_others + d_j_others - d_ij)
        else:
            d_new_others = np.array([])
        
        # Construct the new distance matrix
        new_m = len(others) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others:
            # Fill the block of distances among the 'others'
            D_new[:len(others), :len(others)] = D[np.ix_(others, others)]
            
            # Fill the distances between the new node and the 'others'
            D_new[:len(others), len(others)] = d_new_others
            D_new[len(others), :len(others)] = d_new_others
            # D_new[len(others), len(others)] is 0 by initialization
            
            # Update the clusters list
            # Keep the clusters corresponding to 'others' and add the new one at the end
            new_clusters = [clusters[k] for k in others]
            new_clusters.append(new_cluster_name)
        else:
            new_clusters = [new_cluster_name]
            D_new = np.array([[0.0]])
            
        D = D_new
        clusters = new_clusters
        m = new_m
        
    # Handle the final 3 nodes
    # The structure is (A, B, C) which is unrooted.
    # In Newick, an unrooted tree with 3 leaves is often written as (A,B,C);
    # However, standard Newick is rooted. For 3 leaves, (A,(B,C)); is a valid rooted representation
    # which corresponds to the unrooted topology. The scorer typically handles unrooted trees
    # by looking at splits. (A,B,C) is a valid Newick for a trifurcation or can be interpreted
    # depending on the parser. To be safe and standard, we can output a rooted form for 3 leaves:
    # (cluster0, (cluster1, cluster2));
    # But since we are in the unrooted context, any binary representation works.
    # Let's just join them.
    return "(" + ",".join(clusters) + ");"

def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    D = _jc_distances(sequences, labels)
    
    # Run NJ
    newick = _neighbor_joining(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
