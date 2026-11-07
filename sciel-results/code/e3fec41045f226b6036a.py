"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Global state to track the number of iterations for the cooling schedule
_sa_state = {'iteration': 0, 'initial_temp': 10.0, 'cooling_rate': 0.9}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Simulated Annealing Guided Local Search.
    
    Uses a geometric cooling schedule to determine the penalty strength.
    Early iterations use high penalties to escape local optima broadly,
    while later iterations use lower penalties to fine-tune the solution.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    # Update global state
    # The evaluator calls this function 20 times.
    # We reset the state if we detect a new run (e.g., if iteration exceeds expected count, 
    # though in this single-instance context, it just increments).
    # However, since we don't know when a new instance starts, we assume the 20 calls
    # are sequential for one instance. 
    # To be safe, we can use the edge_n_used max value or just a counter.
    # The problem states "repeats 20 times". 
    # If we run multiple instances, the counter will keep going. 
    # We should reset if we detect a "restart" or just cap the temperature.
    
    # A more robust way: infer the "age" from edge_n_used? 
    # No, edge_n_used accumulates. 
    # Let's just use a simple counter that wraps around or resets if it goes too high.
    # Given 20 iterations, if we see iteration 21, we might be in a new instance.
    # But we don't know the instance boundary. 
    # Let's assume the function is called 20 times per instance.
    # If we call it 40 times total (2 instances), the temperature would be too low for the second instance.
    # We need a way to detect a new instance. 
    # One heuristic: if the current tour length is significantly different from previous? No, we don't have history.
    # Another: if edge_n_used is reset? The problem says "accumulated over the rounds". 
    # It doesn't say it resets between instances. 
    # Wait, "For each instance... repeats 20 times". 
    # Does edge_n_used reset between instances? 
    # "edge_n_used is a symmetric count matrix accumulated over the rounds." 
    # Usually in these tasks, the state is local to the instance run. 
    # Let's assume the function is stateless or that we can detect a new instance.
    # If we can't detect a new instance, a global counter is dangerous.
    
    # Alternative: Use the magnitude of edge_n_used to estimate progress?
    # If edge_n_used is all zeros, it's the first iteration.
    # If max(edge_n_used) is 1, it's likely the second.
    # If max(edge_n_used) is k, it's likely iteration k+1.
    # This is a robust way to determine the "temperature" step without global state!
    
    max_usage = np.max(edge_n_used)
    # max_usage is 0 for iter 1, 1 for iter 2, ..., 19 for iter 20.
    # So iteration index t = max_usage.
    
    T0 = 10.0
    alpha = 0.85  # Cooling factor
    
    # Temperature T_t = T0 * alpha^t
    T = T0 * (alpha ** max_usage)
    
    # Create a copy of the true distance matrix
    perturbed_dist = edge_distance.copy()
    
    # Identify edges in the current tour
    # We apply a penalty to the edges of the current tour.
    # The penalty is proportional to the temperature.
    # penalty_factor = 1.0 + T
    # As T decreases, the penalty becomes smaller, allowing the 2-opt to
    # potentially keep some edges if they are still good, but early on
    # the high T forces a major change.
    
    penalty_factor = 1.0 + T
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        perturbed_dist[u, v] *= penalty_factor
        perturbed_dist[v, u] *= penalty_factor

    return perturbed_dist
# EVOLVE-BLOCK-END
