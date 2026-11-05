import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    deg_safe = np.where(degrees == 0, 1.0, degrees)
    
    # --- Eigenvector centrality via power iteration ---
    # Identifies structurally central nodes in the core
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)
    
    max_iter = 60
    for _ in range(max_iter):
        if time.time() - start_time > deadline - 5:
            break
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
    
    eig_centrality = x.copy()
    
    # Normalize eigenvector centrality to [0, 1]
    if eig_centrality.max() > eig_centrality.min():
        eig_norm = (eig_centrality - eig_centrality.min()) / (eig_centrality.max() - eig_centrality.min())
    else:
        eig_norm = np.zeros(n)
    
    # --- Local edge density (clustering-like metric) ---
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    
    denom = degrees * (degrees - 1.0)
    denom[denom < 1e-10] = 1.0
    rho = np.where(degrees >= 2, two_L / denom, 0.0)
    
    # --- Hub-blocking benefit ---
    # Picking a vertex adjacent to high-degree hubs frees up many other vertices
    neighbor_deg_sum = A @ degrees
    avg_neighbor_deg = neighbor_deg_sum / deg_safe
    
    # Normalize hub-blocking score
    if avg_neighbor_deg.max() > avg_neighbor_deg.min():
        hub_block_norm = (avg_neighbor_deg - avg_neighbor_deg.min()) / (avg_neighbor_deg.max() - avg_neighbor_deg.min())
    else:
        hub_block_norm = np.zeros(n)
    
    # --- Normalize degree ---
    if degrees.max() > degrees.min():
        deg_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    else:
        deg_norm = np.zeros(n)
    
    # --- Iterative refinement of vertex weights ---
    # Base weight: low degree and low eigenvector centrality are more valuable
    base_weights = 1.0 / (degrees + 1.0) * (1.0 - 0.4 * eig_norm)
    base_weights = np.maximum(base_weights, 1e-6)
    
    weights = base_weights.copy()
    
    # Refine weights considering neighbor competition and local structure
    for iteration in range(12):
        if time.time() - start_time > deadline - 3:
            break
        
        neighbor_weight_sum = A @ weights
        
        # Updated weight: base value adjusted for neighbor competition
        # High eigenvector centrality neighbors reduce effective value
        # High local density means neighbors conflict, reducing cost of losing them
        neighbor_eig_sum = A @ eig_centrality
        neighbor_eig_avg = neighbor_eig_sum / deg_safe
        
        weights_new = base_weights / (1.0 + 0.4 * neighbor_weight_sum / (deg_safe + 1.0)) * (1.0 - 0.2 * neighbor_eig_avg)
        weights_new = np.maximum(weights_new, 1e-6)
        
        if np.allclose(weights, weights_new, atol=1e-10):
            weights = weights_new
            break
        weights = weights_new
    
    # --- Final composite score ---
    # Components:
    # 1. Low degree (penalize high degree)
    # 2. Low eigenvector centrality (penalize central nodes) 
    # 3. Local density bonus (neighbors conflict, less costly to lose)
    # 4. Hub-blocking benefit (picking blocks hubs, frees up their other neighbors)
    # 5. Neighbor weight penalty (high-value neighbors are costly to eliminate)
    
    neighbor_value_sum = A @ weights
    neighbor_value_avg = neighbor_value_sum / deg_safe
    
    # Weight the components
    w_deg = 1.0
    w_eig = 0.6
    w_density = 0.3
    w_hub = 0.5
    w_neighbor = 0.4
    
    # Score: higher = selected earlier
    # Penalize: high degree, high eigenvector centrality, high neighbor value
    # Reward: local density (neighbors conflict), hub-blocking
    scores = (
        -w_deg * deg_norm
        - w_eig * eig_norm
        + w_density * rho
        + w_hub * hub_block_norm
        - w_neighbor * neighbor_value_avg
    )
    
    # Safety check
    if time.time() - start_time > deadline:
        return scores.astype(np.float64)
    
    return scores.astype(np.float64)
