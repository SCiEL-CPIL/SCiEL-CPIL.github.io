"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex for greedy independent set decoding.
    Higher score = picked earlier.
    Strategy: reward low-degree vertices that are adjacent to high-degree vertices,
    as picking them blocks high-competition neighbors without much self-conflict.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin: finish well before 120s

    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Convert to numpy array if needed
    adj = np.asarray(adjacency, dtype=np.int8)
    if adj.shape != (n, n):
        # Handle case where adjacency might be given differently
        adj = np.array(adjacency, dtype=np.int8)
        n = adj.shape[0]

    degrees = adj.sum(axis=1).astype(float)
    
    # If graph is empty (no edges), all vertices are independent
    if degrees.max() == 0:
        return np.zeros(n, dtype=float)

    max_deg = degrees.max()
    if max_deg == 0:
        return np.zeros(n, dtype=float)

    # Sum of neighbor degrees for each vertex
    neighbor_deg_sum = (adj @ degrees)  # shape (n,)
    
    # Base score: negative degree (low degree is good)
    score = -degrees
    
    # Bonus: vertices adjacent to high-degree neighbors get a boost
    # This is because picking a low-degree vertex next to high-degree ones
    # blocks those high-degree vertices from competing with others
    if max_deg > 0:
        neighbor_weight = neighbor_deg_sum / max_deg
        # Weight the bonus: higher neighbor degrees = more "value" in blocking them
        score += 0.5 * neighbor_weight
    
    # Additional refinement: penalize vertices in dense neighborhoods
    # Compute for each vertex, the average degree of its neighbors' neighbors (2-hop)
    # This is expensive, so we do a lightweight version:
    # If a vertex's neighbors are highly interconnected, picking it is less "efficient"
    # because those neighbors would have been blocked anyway.
    
    # Compute neighborhood density proxy: for each vertex i, 
    # the number of edges among its neighbors / (deg_i * (deg_i-1) / 2)
    # But this is O(n^3) in worst case. For n=200, let's do it efficiently.
    
    elapsed = time.time() - start_time
    if elapsed < 100.0:  # We have time for more computation
        # Compute 2-hop structure
        # adj2[i,j] = number of common neighbors of i and j
        # For neighborhood density of i: sum over j in N(i) of adj[i,j] * deg_of_neighbors...
        # Simpler: for each vertex i, compute the number of edges within N(i)
        # edges_in_N_i = 0.5 * sum_{j in N(i)} sum_{k in N(i), k != j} adj[j,k]
        #               = 0.5 * (sum_{j in N(i)} (adj @ adj)[i, j] - deg_i)
        
        # adj2 = adj @ adj gives number of length-2 paths from i to j
        # For vertex i, sum over j in N(i) of adj2[i,j] = sum over j in N(i) of (number of common neighbors of i and j)
        # This equals sum over j in N(i) of (number of k such that i-k-j is a path)
        # The number of edges in N(i) is: 0.5 * (sum_{j in N(i)} adj2[i,j] - deg_i)
        # because each edge (j,k) in N(i) is counted twice (once from j to k, once from k to j)
        # and the self-loops (j=j) contribute deg_i
        
        adj2 = adj @ adj  # (n, n), int
        # For each i, sum over j where adj[i,j]=1 of adj2[i,j]
        # Use matrix multiplication: (adj * adj2) @ ones? No.
        # sum_{j in N(i)} adj2[i,j] = (adj * adj2).sum(axis=1)
        neighbor_2hop = (adj * adj2).sum(axis=1).astype(float)
        
        # Number of edges in neighborhood of i:
        # edges_N_i = 0.5 * (neighbor_2hop[i] - degrees[i])
        # Because adj2[i,j] for j in N(i) counts paths i-k-j, and for k=j it's a self-loop (1 if j in N(i))
        # Actually adj2[i,i] = deg_i (number of length-2 paths from i to i, which are i-k-i)
        # For j in N(i), adj2[i,j] = number of common neighbors of i and j
        # The number of edges among N(i) = 0.5 * sum_{j in N(i)} (number of common neighbors of i and j that are in N(i))
        # This is getting complex. Let's use a simpler proxy.
        
        # Simpler: use the average degree of neighbors as a density proxy
        # If neighbors have high average degree, the neighborhood is "dense"
        avg_neighbor_deg = np.zeros(n, dtype=float)
        for i in range(n):
            if degrees[i] > 0:
                avg_neighbor_deg[i] = neighbor_deg_sum[i] / degrees[i]
            else:
                avg_neighbor_deg[i] = 0
        
        # Penalize vertices whose neighbors have high average degree (dense neighborhood)
        # because picking such a vertex doesn't "save" many potential picks
        score -= 0.3 * avg_neighbor_deg / max_deg
        
    # Final score: higher is better (picked earlier)
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
