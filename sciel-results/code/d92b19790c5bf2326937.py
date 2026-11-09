"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using Graph Laplacian Spectral Clustering.
    
    The heuristic is designed to encourage edges within dense spectral clusters (low-frequency)
    and suppress edges that cross natural partition boundaries (high-frequency).
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # 1. Construct a weighted adjacency matrix for spectral analysis
    # We use inverse distance as weights, so closer nodes have higher connectivity.
    # Add a small epsilon to avoid division by zero for self-loops or identical coordinates.
    eps = 1e-6
    safe_dist = distance_matrix + eps
    
    # Weights: Higher weight for shorter distances.
    # Using a power law to sharpen the difference between close and far nodes.
    weights = 1.0 / (safe_dist ** 2)
    
    # Set diagonal to 0 for the adjacency matrix used in Laplacian
    np.fill_diagonal(weights, 0.0)
    
    # 2. Compute the Graph Laplacian L = D - A
    # D is the degree matrix (diagonal)
    degrees = np.sum(weights, axis=1)
    D = np.diag(degrees)
    L = D - weights
    
    # 3. Compute Eigenvectors
    # We are interested in the Fiedler vector (2nd smallest eigenvalue).
    # numpy.linalg.eigh returns eigenvalues in ascending order.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The first eigenvector (index 0) corresponds to lambda=0 (constant vector).
        # The second eigenvector (index 1) is the Fiedler vector.
        fiedler_vec = eigenvectors[:, 1]
        
        # Normalize the Fiedler vector to have zero mean and unit variance for consistent scaling
        fiedler_mean = np.mean(fiedler_vec)
        fiedler_std = np.std(fiedler_vec)
        if fiedler_std > 1e-9:
            fiedler_norm = (fiedler_vec - fiedler_mean) / fiedler_std
        else:
            fiedler_norm = fiedler_vec
            
    except Exception:
        # Fallback to simple inverse distance if spectral decomposition fails
        fiedler_norm = np.zeros(n)
        
    # 4. Construct the Heuristic Matrix
    # Base heuristic: Inverse distance (standard ACO heuristic)
    base_heuristic = 1.0 / (safe_dist ** 1.5)
    
    # Spectral affinity component:
    # We want to boost edges where the Fiedler values are similar (same cluster).
    # Difference in Fiedler values: d_f[i, j] = |fiedler_norm[i] - fiedler_norm[j]|
    # Smaller difference -> Higher affinity.
    # Use an exponential decay to create a smooth surrogate for the "same cluster" indicator.
    diff_fiedler = np.abs(fiedler_norm[:, np.newaxis] - fiedler_norm[np.newaxis, :])
    
    # Scale factor for the exponential decay. 
    # A smaller sigma makes the clustering more aggressive (sharper boundary).
    # A larger sigma makes it more diffuse.
    # Based on the distribution of differences, a median-based scale is robust.
    median_diff = np.median(diff_fiedler[diff_fiedler > 1e-9]) if np.any(diff_fiedler > 1e-9) else 1.0
    sigma = 0.5 * median_diff
    
    # Affinity score: 1.0 if same cluster, decays as difference increases.
    spectral_affinity = np.exp(-diff_fiedler ** 2 / (2 * sigma ** 2))
    
    # Combine base heuristic with spectral affinity.
    # We multiply them so that both proximity and cluster membership are required for high attractiveness.
    # This effectively suppresses edges that are short but cross cluster boundaries (if they exist)
    # and boosts edges that are within the same cluster.
    heuristic = base_heuristic * spectral_affinity
    
    # 5. Demand-aware adjustment
    # Slightly prefer nodes with higher demand to be visited, but keep it mild
    # to not disrupt the spectral structure too much.
    max_demand = np.max(demands)
    if max_demand > 0:
        demand_norm = demands / max_demand
    else:
        demand_norm = demands
        
    # Mild demand boost: 1.0 to 1.05
    demand_factor = 1.0 + 0.05 * demand_norm
    demand_factor[0] = 1.0 # Depot neutral
    
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # 6. Cleanup
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Set diagonal to 0 (or very small) as ants don't visit themselves
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
