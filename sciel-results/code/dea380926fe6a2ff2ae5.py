"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign scores to vertices for greedy independent set selection.
    Higher score = selected earlier.
    
    Strategy: Use a combination of:
    1. Low degree (prefer peripheral nodes)
    2. Low eigenvector centrality (prefer nodes not in dense cores)
    3. Normalized neighbor hub-blocking benefit (prefer nodes adjacent to high-degree hubs,
       because picking them 'blocks' the hub and frees up many of the hub's other neighbors)
    
    We compute a composite score that balances these factors.
    """
    start_time = time.time()
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Degree of each vertex
    deg = A.sum(axis=1)
    
    # Avoid division by zero
    deg_safe = np.where(deg == 0, 1.0, deg)
    
    # --- Eigenvector centrality via power iteration ---
    # We want the principal eigenvector of A.
    # Initialize with uniform vector.
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() - start_time > 100:  # safety margin
            break
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
    
    # Eigenvector centrality: higher means more central
    eig_centrality = x.copy()
    
    # --- Normalize eigenvector centrality to [0, 1] range for easier combination ---
    if eig_centrality.max() > eig_centrality.min():
        eig_norm = (eig_centrality - eig_centrality.min()) / (eig_centrality.max() - eig_centrality.min())
    else:
        eig_norm = np.zeros(n)
    
    # --- Normalize degree to [0, 1] range ---
    if deg.max() > deg.min():
        deg_norm = (deg - deg.min()) / (deg.max() - deg.min())
    else:
        deg_norm = np.zeros(n)
    
    # --- Hub-blocking benefit ---
    # For each vertex i, the benefit of picking it is related to how many of its neighbors
    # have high degree (i.e., picking i blocks a hub, which frees up the hub's other neighbors).
    # A simple measure: sum over neighbors j of (deg[j] / avg_neighbor_degree)
    # This captures that neighbors of high-degree hubs are more "valuable" to block.
    
    # For each vertex i, compute the average degree of its neighbors
    # sum of degrees of neighbors = A @ deg
    neighbor_deg_sum = A @ deg
    avg_neighbor_deg = neighbor_deg_sum / deg_safe
    
    # Hub-blocking score: for vertex i, how much "hub value" it blocks
    # We use: sum over neighbors j of deg[j] / avg_neighbor_deg[i]
    # This is higher when i's neighbors are hubs relative to i's neighborhood
    # But we want to PREFER vertices that block hubs, so higher score = better
    # However, we also want low degree, so we need to balance.
    
    # Let's define: hub_block_score[i] = (A @ deg) / deg_safe[i] = avg_neighbor_deg[i]
    # This is the average degree of i's neighbors. Higher means i is adjacent to hubs.
    # Picking such a vertex blocks those hubs, which is good.
    
    # Normalize hub_block_score to [0, 1]
    hub_block = avg_neighbor_deg.copy()
    if hub_block.max() > hub_block.min():
        hub_block_norm = (hub_block - hub_block.min()) / (hub_block.max() - hub_block.min())
    else:
        hub_block_norm = np.zeros(n)
    
    # --- Composite score ---
    # We want:
    # - Low degree (penalize high degree)
    # - Low eigenvector centrality (penalize central nodes)
    # - High hub-blocking benefit (reward blocking hubs)
    
    # Let's use a weighted combination. The weights are chosen based on intuition:
    # - Degree is a strong local signal
    # - Eigenvector centrality captures global structure
    # - Hub-blocking captures the strategic value of blocking hubs
    
    # Score = -w1 * deg_norm - w2 * eig_norm + w3 * hub_block_norm
    # We want higher score = selected earlier.
    
    w1 = 1.0  # weight for degree
    w2 = 0.5  # weight for eigenvector centrality
    w3 = 0.5  # weight for hub-blocking
    
    score = -w1 * deg_norm - w2 * eig_norm + w3 * hub_block_norm
    
    # Final safety check
    if time.time() - start_time > 110:
        # If we're running out of time, just return the score we have
        pass
    
    return score.astype(np.float64)
# EVOLVE-BLOCK-END
