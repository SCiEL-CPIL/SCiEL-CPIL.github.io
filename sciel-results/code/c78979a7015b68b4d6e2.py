"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Combine spectral initialization with combinatorial ordering search (Simulated Annealing).
    1. Initialize the vertex ordering using spectral relaxation (Fiedler vector) combined with degree heuristics.
    2. Treat the ordering as a discrete state space.
    3. Use Simulated Annealing to swap vertices in the ordering to escape local optima.
    4. Return the scores corresponding to the best ordering found.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Fast Greedy Simulator
    def simulate_greedy(order):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            # Mark neighbors as unavailable
            available[adj_bool[v]] = False
            
            # Early exit if no vertices available
            if not available.any():
                break
                
        return selected

    # Spectral Initialization
    # Use the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue of the Laplacian)
    # to identify a "cut" in the graph. Vertices on the same side of the cut are more likely to be independent.
    # We combine this with degree information.
    
    # Compute the Laplacian matrix L = D - A
    L = np.diag(degree) - adj_bool.astype(float)
    
    # Compute eigenvalues and eigenvectors
    # For n=200, this is fast enough.
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue (eigh returns in ascending order)
        # The second smallest eigenvalue corresponds to the Fiedler vector
        if n > 1:
            fiedler_vec = eigvecs[:, 1]
        else:
            fiedler_vec = np.zeros(n)
    except Exception:
        fiedler_vec = np.random.normal(0, 1, n)
    
    # Normalize fiedler vector
    fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
    if np.std(fiedler_vec) > 0:
        fiedler_vec = fiedler_vec / np.std(fiedler_vec)
    
    # Combine spectral info with degree
    # We want to prioritize vertices that are:
    # 1. Low degree (classic heuristic)
    # 2. On the "same side" of the spectral cut (less likely to conflict)
    
    # Create an initial score for sorting
    # Lower score = higher priority (selected first)
    # We use a combination of degree and fiedler vector value
    
    # Normalize degree and fiedler vector
    max_deg = degree.max() if degree.max() > 0 else 1.0
    degree_norm = degree / max_deg
    
    # The fiedler vector values range roughly from -1 to 1
    # We want vertices with similar fiedler values to be selected together?
    # Actually, the Fiedler vector partitions the graph. 
    # A common heuristic is to select vertices based on the sign of the Fiedler vector.
    # Let's create a score that combines degree and fiedler value.
    
    # Score = degree_norm + alpha * |fiedler_vec|
    # Vertices with low degree and low |fiedler| (near the cut) might be good?
    # Or vertices with low degree and high |fiedler| (far from cut)?
    
    # Let's try: Score = degree_norm + 0.5 * (1 - np.abs(fiedler_vec))
    # This prefers low degree and vertices far from the cut (strong spectral signature)
    
    initial_score = degree_norm + 0.5 * (1 - np.abs(fiedler_vec))
    
    # Add small random noise to break ties
    initial_score += np.random.normal(0, 0.01, n)
    
    # Get initial order
    current_order = np.argsort(initial_score, kind='stable')
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    cooling = 0.9995  # Cooling factor
    
    # Main Search Loop
    max_iters = 200000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate Neighbor: Swap two random positions in the order
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        
        if idx1 == idx2:
            continue
            
        new_order = current_order.copy()
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        
        new_size = simulate_greedy(new_order)
        
        delta = new_size - current_size
        
        # Acceptance Criteria
        if delta > 0:
            # Always accept improvement
            current_order = new_order
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_order = new_order
                current_size = new_size
        
        # Cool down
        T *= cooling
        
        # Restart mechanism
        if T < 0.01:
            # Probability to restart from best found or a new spectral heuristic
            if np.random.random() < 0.1:
                # Restart from best
                current_order = best_order.copy()
                current_size = best_size
                T = 0.5  # Reheat slightly
            elif np.random.random() < 0.1:
                # Restart from new spectral heuristic
                initial_score = degree_norm + 0.5 * (1 - np.abs(fiedler_vec))
                initial_score += np.random.normal(0, 0.01, n)
                current_order = np.argsort(initial_score, kind='stable')
                current_size = simulate_greedy(current_order)
                T = 0.5

    # Convert best_order to scores
    scores = np.zeros(n)
    # The vertex at best_order[i] should have rank i (0 is best)
    # We want score to be inversely proportional to rank.
    # score[best_order[i]] = n - i
    scores[best_order] = n - np.arange(n)
    
    return scores
# EVOLVE-BLOCK-END
