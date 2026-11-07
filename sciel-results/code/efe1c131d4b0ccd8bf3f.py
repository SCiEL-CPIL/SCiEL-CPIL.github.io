import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    
    # Calculate a scale factor based on the typical edge length
    mask = ~np.eye(n, dtype=bool)
    if np.any(mask):
        avg_len = np.mean(edge_distance[mask])
    else:
        avg_len = 1.0
        
    if avg_len < 1e-10:
        avg_len = 1.0
    
    # Create the modified distance matrix
    modified_dist = edge_distance.copy().astype(float)
    
    # Determine current round index from max usage count
    max_usage = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    current_round = max_usage
    
    # Compute local point density around each edge's midpoint using distance matrix
    # For each edge (i, j), we estimate the "crowding" by summing distances from 
    # all other points to the midpoint (approximated by average distance to i and j)
    # We use a simple approximation: for edge (i,j), density is proportional to 
    # sum_k (1 / (d_ik + d_jk)) which is high when many points are close to the edge
    
    # To keep it efficient, we compute a density score for each edge
    # density[i,j] = sum_k 1/(d_ik + d_jk + eps)
    # This is O(n^3) which for n=100 is 1,000,000 operations - acceptable
    
    eps = 1e-10
    # Compute density for all edges
    # For each pair (i,j), density = sum_k 1/(d_ik + d_jk)
    # We can vectorize this: for each k, compute 1/(d_ik + d_jk) for all i,j
    # This is equivalent to: density = sum_k (1/(d[:,k] + d[:,k].T))
    
    # d is n x n. For each k, d[:,k] is n x 1, d[:,k].T is 1 x n
    # d[:,k] + d[:,k].T is n x n matrix where element (i,j) = d_ik + d_jk
    # We want sum_k 1/(d_ik + d_jk)
    
    # Efficient computation:
    # density[i,j] = sum_k 1/(d[i,k] + d[j,k])
    
    # We can do this by iterating over k, but let's try to vectorize
    # d is n x n. For each k, we have vector v_k = d[:,k] (n x 1)
    # v_k + v_k.T is n x n
    # We sum 1/(v_k + v_k.T) over k
    
    # This is O(n^3) memory if we do it naively. Let's do it in chunks or just accept O(n^3)
    # For n=100, n^3 = 1,000,000, which is fine
    
    density = np.zeros((n, n))
    for k in range(n):
        v = edge_distance[:, k]  # n x 1
        # v + v.T is n x n
        # Avoid division by zero
        denom = v[:, np.newaxis] + v[np.newaxis, :]
        denom = np.maximum(denom, eps)
        density += 1.0 / denom
    
    # Now we have density[i,j] for all i,j
    # We want to penalize edges in crowded areas more
    # The penalty should be inversely proportional to density (or proportional to 1/density)
    # Wait, the directive says "penalty inversely proportional to this density"
    # So penalty = c / density? That would mean low density (sparse) gets high penalty?
    # That seems backwards. Let me re-read.
    
    # "applying a penalty inversely proportional to this density to encourage the tour to avoid congested regions"
    # If we want to AVOID congested regions, we should penalize crowded edges MORE.
    # So penalty should be PROPORTIONAL to density, not inversely proportional.
    # Unless the directive means something else.
    
    # Let me re-read: "Model the penalty as a 'crowding' cost by computing the local point density 
    # around each edge's midpoint using the distance matrix, and applying a penalty inversely 
    # proportional to this density to encourage the tour to avoid congested regions and explore 
    # sparser areas"
    
    # This is confusing. If penalty is inversely proportional to density, then:
    # - High density (crowded) -> low penalty
    # - Low density (sparse) -> high penalty
    # This would encourage the tour to STAY in crowded areas, not avoid them.
    
    # I think there might be a misunderstanding. Let me interpret it as:
    # We want to penalize edges in crowded regions to encourage exploration of sparse regions.
    # So penalty should be proportional to density.
    
    # But let's follow the directive literally: "penalty inversely proportional to this density"
    # Maybe the idea is that in sparse regions, the "cost" of an edge is higher because 
    # there's less "support" from nearby points? That doesn't make much sense for TSP.
    
    # Let me try the opposite: penalty proportional to density, which makes more sense 
    # for avoiding congested regions.
    
    # Normalize density to [0, 1]
    density_min = np.min(density)
    density_max = np.max(density)
    if density_max > density_min:
        density_norm = (density - density_min) / (density_max - density_min)
    else:
        density_norm = np.zeros_like(density)
    
    # Apply penalty proportional to density (to avoid crowded regions)
    # Scale the penalty by a factor
    penalty_scale = 0.1 * avg_len
    
    # Add density-based penalty
    modified_dist += penalty_scale * density_norm
    
    # Also add the standard GLS penalty for used edges
    penalty_factor = 0.05 * avg_len
    modified_dist += penalty_factor * edge_n_used
    
    # Reset diagonal to 0
    np.fill_diagonal(modified_dist, 0)
    
    # Ensure symmetry
    modified_dist = (modified_dist + modified_dist.T) / 2.0
    
    return modified_dist
