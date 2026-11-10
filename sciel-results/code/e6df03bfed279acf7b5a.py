import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # 1. Construct the graph weight matrix W
    # Edge weight is inversely proportional to residual space if they fit
    # W_ij = 1 / (C - d_i - d_j) if d_i + d_j <= C, else 0
    # We use a small epsilon to avoid division by zero if d_i+d_j == C (residual 0)
    # In that case, the weight should be very high.
    
    sum_pair = demands[:, None] + demands[None, :]
    residual = capacity - sum_pair
    
    # Initialize W with 0
    W = np.zeros((n, n))
    
    # Mask for feasible pairs (including diagonal? Usually self-loops are 0 in Laplacian context)
    # The task says "placing item j right after item i". i and j can be same? 
    # "places a random fitting item... picks a fitting item". 
    # Usually an item cannot be placed twice. 
    # But the heuristic matrix is N x N. 
    # If i == j, it means placing the same item twice? 
    # The ACO logic: "picks a fitting item". It likely maintains a list of available items.
    # So H_ii might not be used if item i is already placed.
    # However, for safety, let's define W_ii = 0.
    
    feasible = (sum_pair <= capacity) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    
    # Calculate weights for feasible pairs
    # residual is positive for feasible pairs (strictly > 0 if we exclude exact fit? No, <= means residual >= 0)
    # If residual == 0, weight is inf. We cap it.
    eps = 1e-6
    valid_residual = np.where(feasible, residual, 1.0) # Avoid div by 0 in calc
    weights = 1.0 / (valid_residual + eps)
    
    # Set non-feasible to 0
    W = np.where(feasible, weights, 0.0)
    
    # 2. Compute Laplacian L = D - W
    # Degree matrix D
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    L = D - W
    
    # 3. Compute Eigenvalues and Eigenvectors
    # We need the Fiedler vector (2nd smallest eigenvalue)
    # np.linalg.eigh is for symmetric matrices, which L is.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic if eigen decomposition fails
        # Return baseline-like heuristic
        pair = demands[:, None] + demands[None, :]
        return np.where(pair <= capacity, pair / capacity, 1e-9)

    # Sort eigenvalues and vectors
    # eigh returns in ascending order
    # eigenvalues[0] is usually 0 (for connected graph)
    # eigenvalues[1] is the Fiedler value
    # eigenvectors[:, 1] is the Fiedler vector
    
    if len(eigenvalues) < 2:
        # Fallback
        pair = demands[:, None] + demands[None, :]
        return np.where(pair <= capacity, pair / capacity, 1e-9)

    fiedler_vec = eigenvectors[:, 1]
    
    # 4. Compute Heuristic Matrix
    # H_ij = v_i * v_j
    # We want to encourage positive products.
    # If v_i * v_j is negative, it becomes 1e-9.
    
    H = np.outer(fiedler_vec, fiedler_vec)
    
    # Normalize H to be in a reasonable range, e.g., max value 1.0
    # This helps with the pheromone scaling (which starts at 1)
    max_h = np.max(H)
    if max_h > 0:
        H = H / max_h
    else:
        # If all zero or negative, fallback
        pair = demands[:, None] + demands[None, :]
        return np.where(pair <= capacity, pair / capacity, 1e-9)

    # 5. Apply Feasibility Mask
    # If pair is not feasible, H_ij = 1e-9
    # If H_ij <= 0, H_ij = 1e-9 (handled by "not positive")
    
    # The prompt says: "Heuristic entries that are not positive become 1e-9"
    # So we first apply the positivity rule, then the feasibility rule?
    # Or just combine them.
    
    # Feasible mask
    feasible_mask = (sum_pair <= capacity)
    
    # Apply positivity: H = np.where(H > 0, H, 1e-9)
    # Apply feasibility: H = np.where(feasible_mask, H, 1e-9)
    
    # Combined:
    H = np.where(feasible_mask & (H > 0), H, 1e-9)
    
    return H
