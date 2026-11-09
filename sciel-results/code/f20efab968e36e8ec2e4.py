"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_mis_size(adjacency, scores):
    """Run the fixed greedy decoder and return the independent set size."""
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    
    order = np.argsort(-scores, kind='stable')
    
    count = 0
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        # Mark neighbors as unavailable
        neighbors = np.where(adjacency[v])[0]
        available[neighbors] = False
        available[v] = False
    return count


def priority(adjacency):
    """
    Compute vertex scores for the greedy MIS decoder.
    
    Strategy: Use a local search / simulated annealing approach to find
    a good ordering (score vector) that maximizes the greedy MIS size.
    
    The key insight: we want to select vertices that are "easy" to include
    (low degree, isolated in the remaining graph) while avoiding selecting
    high-degree hubs early.
    
    We'll use an iterative refinement approach:
    1. Start with a reasonable initial score (e.g., based on degree and 
       neighborhood structure)
    2. Iteratively adjust scores to improve the greedy outcome
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: 120s limit
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Initial score: combine multiple heuristics
    # 1. Low degree is good (prefer leaves/isolated vertices)
    # 2. High "independence potential": vertices whose neighbors have high degree
    #    (selecting them removes many edges from consideration)
    
    # Sum of degrees of neighbors (higher means selecting this vertex 
    # "protects" more edges from being in the independent set problem)
    neighbor_degree_sums = adjacency.dot(degrees)
    
    # Number of common neighbors between pairs - use as proxy for 
    # how "clique-like" the neighborhood is
    # For each vertex, count how many pairs of its neighbors are also connected
    # This is expensive, so use a simpler proxy: degree of degree
    
    # Initial score: prefer low degree, high neighbor-degree-sum
    # Normalize to avoid scale issues
    score = -degrees + 0.5 * neighbor_degree_sums
    
    # Normalize
    if score.std() > 1e-10:
        score = (score - score.mean()) / (score.std() + 1e-10)
    
    # Evaluate initial score
    best_score_vec = score.copy()
    best_mis_size = _greedy_mis_size(adjacency, best_score_vec)
    
    # Local search: perturb scores and keep improvements
    # Use simulated annealing style: accept worse solutions with some probability
    rng = np.random.RandomState(42)
    
    current_score_vec = best_score_vec.copy()
    current_mis_size = best_mis_size
    
    # Temperature schedule for simulated annealing
    initial_temp = 1.0
    cooling_rate = 0.995
    
    max_iterations = 5000  # Cap iterations to stay within time
    
    for iteration in range(max_iterations):
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Perturb: add random noise to scores
        noise = rng.normal(0, 0.5, size=n)
        candidate_score_vec = current_score_vec + noise
        
        # Evaluate
        candidate_mis_size = _greedy_mis_size(adjacency, candidate_score_vec)
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        delta = current_mis_size - candidate_mis_size
        if delta < 0:
            # Candidate is better
            current_score_vec = candidate_score_vec
            current_mis_size = candidate_mis_size
            if current_mis_size > best_mis_size:
                best_mis_size = current_mis_size
                best_score_vec = current_score_vec.copy()
        else:
            # Candidate is worse or equal
            temp = initial_temp * (cooling_rate ** iteration)
            if temp > 1e-6:
                accept_prob = np.exp(-delta / (temp + 1e-10))
                if rng.random() < accept_prob:
                    current_score_vec = candidate_score_vec
                    current_mis_size = candidate_mis_size
        
        # If we've been stuck for a while, do a larger perturbation
        if iteration > 0 and iteration % 500 == 0:
            # Reset current to best and do a larger jump
            current_score_vec = best_score_vec.copy()
            current_mis_size = best_mis_size
    
    # Final polish: try small perturbations around the best solution
    if time.time() < deadline - 2.0:
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            noise = rng.normal(0, 0.1, size=n)
            candidate_score_vec = best_score_vec + noise
            candidate_mis_size = _greedy_mis_size(adjacency, candidate_score_vec)
            if candidate_mis_size > best_mis_size:
                best_mis_size = candidate_mis_size
                best_score_vec = candidate_score_vec.copy()
    
    return best_score_vec
# EVOLVE-BLOCK-END
