import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    Uses a time-decaying cooling schedule to model expected pheromone reinforcement,
    combined with demand-awareness and structural distance bias.
    """
    start_time = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    
    # Distance from each node to the depot (node 0)
    depot_dist = d[:, 0].copy()
    
    # Demand-aware structural heuristic with cooling schedule:
    # We simulate the expected pheromone trajectory and create a static heuristic
    # that encodes the average attractiveness over the search.
    
    # Base structural cost: d[i,j] + beta * d[j,0]
    # This accounts for the cost of visiting j and returning to depot.
    beta = 0.5
    
    # Demand penalty: high-demand customers should be visited earlier (when capacity is available)
    # but the heuristic should not overly penalize them as they are necessary.
    # We use a mild demand-awareness term.
    gamma = 0.2
    
    # Cooling schedule: early in the search, pheromone has less influence,
    # so the heuristic should be more dominant. Later, pheromone dominates.
    # We model the "effective" heuristic as a weighted average over time.
    
    # Simulate a simplified cooling: the heuristic exponent decays over time.
    # In ACO, probability ~ pheromone^alpha * heuristic^beta
    # As pheromone builds up, the relative influence of heuristic decreases.
    # We approximate this by computing heuristic values with different "sharpness"
    # and averaging them.
    
    # Number of time steps to simulate for the cooling effect
    n_steps = 10
    
    # For each time step t, the effective "temperature" or sharpness of the heuristic
    # decreases, meaning the heuristic becomes less selective as search progresses.
    # We model this by scaling the denominator: smaller denominator = more attractive.
    
    # At early times, the heuristic should be more selective (sharper)
    # At later times, it should be flatter (less selective)
    
    # We'll compute a weighted average of heuristic matrices with different sharpness levels
    # The weights reflect the probability of being at each time step in the search
    
    # Base denominator: d[i,j] + beta * d[j,0] + gamma * (demand[j]/capacity)
    demand_factor = (demands / capacity)[None, :]  # shape (1, n)
    base_denom = d + beta * depot_dist[None, :] + gamma * demand_factor
    
    # For each time step, we modify the denominator to reflect the cooling
    # Early: sharper heuristic (smaller effective distances for good edges)
    # Late: flatter heuristic (larger effective distances, less discrimination)
    
    # We use a power transformation: denom^p where p < 1 makes it flatter
    # p=1 is the base case, p=0.5 is flatter, p=2 is sharper
    
    # Cooling schedule: p_t decreases from p_start to p_end over the search
    p_start = 1.0  # initial sharpness
    p_end = 0.7    # final sharpness (flatter)
    
    # Exponential decay of sharpness
    # p(t) = p_end + (p_start - p_end) * exp(-t/tau)
    tau = 5.0  # time constant
    
    # Compute weighted average of heuristic matrices
    # Weights are proportional to the time spent at each sharpness level
    # In a 30-iteration search, we sample 10 time points
    
    heuristic_sum = np.zeros_like(base_denom)
    weight_sum = 0.0
    
    for t in range(n_steps):
        # Time fraction from 0 to 1
        frac = t / (n_steps - 1)
        
        # Sharpness at this time
        p_t = p_end + (p_start - p_end) * np.exp(-frac * 3.0)
        
        # Modified denominator: base_denom^p_t
        # For p_t < 1, this compresses large values, making the heuristic flatter
        # For p_t > 1, this expands large values, making it sharper
        modified_denom = np.power(base_denom + 1e-6, p_t)
        
        # Heuristic is inverse of modified denominator
        h_t = 1.0 / (modified_denom + 1e-6)
        
        # Weight: uniform over time steps for simplicity
        # In practice, early steps matter more for the initial search direction
        w_t = 1.0 / (1.0 + 0.5 * frac)  # slight bias toward early times
        
        heuristic_sum += w_t * h_t
        weight_sum += w_t
    
    # Normalize
    heuristic = heuristic_sum / weight_sum
    
    # Set diagonal to 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Replace non-positive or non-finite with 1e-9
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return the current result immediately
        pass
    
    return heuristic
