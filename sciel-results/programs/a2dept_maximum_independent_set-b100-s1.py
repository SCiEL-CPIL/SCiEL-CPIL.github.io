"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _greedy_size(A, scores, n):
    """Run greedy decoder: pick vertices in descending score, skip neighbors of picked."""
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if picked[v] or blocked[v]:
            continue
        picked[v] = True
        count += 1
        # Block neighbors of v
        nbrs = np.where(A[v] > 0.5)[0]
        blocked[nbrs] = True
    return count


def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    Higher score = picked earlier.
    
    Strategy: Simulate the greedy process to estimate vertex "value".
    For each vertex, estimate how much the IS size improves if we force-pick it
    early. Use multiple random perturbations of a base heuristic and keep the
    best-scoring ordering.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=np.float64)
    deg = A.sum(axis=1)
    
    # Base heuristic: low degree + high neighbor degree - blocking cost
    safe_deg = np.where(deg > 0, deg, 1.0)
    mean_neighbor_deg = (A @ deg) / safe_deg
    mean_neighbor_deg = np.where(deg > 0, mean_neighbor_deg, 0.0)
    
    pickability = 1.0 / (1.0 + deg)
    blocking_cost = A @ pickability
    mean_pickability = blocking_cost / safe_deg
    mean_pickability = np.where(deg > 0, mean_pickability, 0.0)
    
    # Base score: combination of structural features
    base_score = -deg + 0.8 * mean_neighbor_deg - 1.2 * mean_pickability
    
    # Evaluate base score
    best_size = _greedy_size(A, base_score, n)
    best_score = base_score.copy()
    
    # Try several perturbations / alternative heuristics
    rng = np.random.RandomState(42)
    
    # Alternative 1: pure degree-based
    alt1 = -deg.copy()
    sz1 = _greedy_size(A, alt1, n)
    if sz1 > best_size:
        best_size = sz1
        best_score = alt1.copy()
    
    # Alternative 2: degree with neighbor degree emphasis
    alt2 = -deg + 2.0 * mean_neighbor_deg
    sz2 = _greedy_size(A, alt2, n)
    if sz2 > best_size:
        best_size = sz2
        best_score = alt2.copy()
    
    # Alternative 3: random tie-breaking variations of base
    for trial in range(50):
        if time.time() > deadline:
            break
        noise = rng.normal(0, 0.5, size=n)
        perturbed = base_score + noise
        sz = _greedy_size(A, perturbed, n)
        if sz > best_size:
            best_size = sz
            best_score = perturbed.copy()
    
    # Alternative 4: simulate "value" by estimating marginal contribution
    # For each vertex v, estimate: if v is picked, how many other vertices become unavailable?
    # And how many of those would have been picked anyway?
    # Use a simple proxy: value(v) = 1 - (number of "would-be-picked" neighbors)
    # Approximate "would-be-picked" by low-degree neighbors
    
    # Try a value-based approach: score by estimated marginal gain
    # Marginal gain of picking v ≈ 1 - P(neighbors would be picked)
    # P(neighbor u picked) ≈ 1/(1+deg(u)) (low degree = more likely picked)
    # So marginal gain ≈ 1 - sum_{u in N(v)} [1/(1+deg(u))] / (1 + sum_{u in N(v)} [1/(1+deg(u))])
    # Simplify: use 1 - mean_pickability as a value proxy
    
    value_score = 1.0 - mean_pickability
    # Combine with degree penalty
    alt3 = value_score - 0.3 * deg
    sz3 = _greedy_size(A, alt3, n)
    if sz3 > best_size:
        best_size = sz3
        best_score = alt3.copy()
    
    # Alternative 5: iterative refinement - simulate greedy, identify "critical" vertices
    # that are picked late but could be picked earlier to improve the set
    # Use the greedy order to identify which vertices are "wasted" (blocked but would have been useful)
    
    # Try: score = -deg + small random perturbation, multiple restarts
    for trial in range(100):
        if time.time() > deadline:
            break
        # Mix of base and random
        w = rng.uniform(0.5, 1.0)
        noise = rng.normal(0, 1.0, size=n)
        perturbed = w * base_score + (1 - w) * noise
        sz = _greedy_size(A, perturbed, n)
        if sz > best_size:
            best_size = sz
            best_score = perturbed.copy()
    
    # Final: try to improve by swapping low-value picked vertices with high-value blocked ones
    # This is a local search on the ordering
    if time.time() < deadline - 5.0:
        # Get current greedy order
        order = np.argsort(-best_score, kind='stable')
        # Try moving some vertices earlier in the order
        for swap_trial in range(200):
            if time.time() > deadline:
                break
            # Pick a random position to perturb
            pos = rng.randint(0, n - 1)
            if pos >= n - 1:
                continue
            # Swap positions pos and pos+1 (or a nearby position)
            new_order = order.copy()
            swap_pos = min(pos + rng.randint(1, 10), n - 1)
            new_order[pos], new_order[swap_pos] = new_order[swap_pos], new_order[pos]
            
            # Convert order back to scores (inverse rank)
            new_scores = np.zeros(n)
            for rank, v in enumerate(new_order):
                new_scores[v] = n - rank
            
            sz = _greedy_size(A, new_scores, n)
            if sz > best_size:
                best_size = sz
                best_score = new_scores.copy()
    
    return best_score
# EVOLVE-BLOCK-END
