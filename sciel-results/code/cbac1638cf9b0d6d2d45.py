"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = (adj > 0.5).astype(np.int8)
    degrees = adj_bool.sum(axis=1).astype(np.float64)
    
    # Precompute neighbor lists for fast access in local search
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_scores(scores):
        """
        Compute the size of the greedy independent set given a score vector.
        The decoder picks vertices in descending score order.
        """
        # Get order of vertices by descending score
        # Using stable sort to ensure deterministic behavior if scores are tied
        order = np.argsort(-scores, kind='stable')
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                # Mark neighbors as blocked
                # Using boolean indexing for speed
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    # --- Step 1: Generate Base Scores using Structural Heuristics ---
    
    # Heuristic 1: Spectral Peripherality
    # Vertices with low centrality are better for MIS.
    # We use the inverse of the Fiedler vector (2nd smallest eigenvector of Laplacian) 
    # or simply the eigenvector corresponding to the smallest eigenvalue (which is constant for connected graphs, so not useful).
    # Better: Use the eigenvector of the adjacency matrix corresponding to the smallest eigenvalue? 
    # No, let's use the "coreness" or "peripherality" based on the Laplacian.
    # Actually, a simpler proxy for peripherality is the inverse of the PageRank or just high distance from center.
    # Let's use the smallest eigenvector of the Laplacian (Fiedler vector). 
    # In a connected graph, the Fiedler vector components indicate "side" of a cut.
    # Vertices with small absolute values in the Fiedler vector are near the "center" of the cut?
    # Actually, for MIS, we want vertices that are "isolated" from each other.
    # A known heuristic is that vertices with low degree in the *square* of the graph (distance-2 neighbors) are good.
    # Let's compute distance-2 degrees.
    
    # Efficient computation of distance-2 degrees:
    # A^2[i, j] = number of common neighbors.
    # Deg2[i] = sum(A^2[i, :]) - deg[i] (if self-loops not present, which they aren't in ER)
    # A^2 = adj_bool @ adj_bool
    try:
        A2 = adj_bool @ adj_bool
        deg2 = A2.sum(axis=1) - degrees # Subtract self-loops (diagonal of A2 is degree)
        # Penalize high distance-2 degree
        score_periph = -deg2
    except Exception:
        score_periph = -degrees

    # Heuristic 2: Maximal Matching Unmatched Vertices
    # Greedily find a maximal matching. Vertices not in the matching are good MIS candidates.
    # This is a heuristic because the matching is not necessarily minimum, but unmatched vertices
    # in *any* maximal matching form an independent set (wait, no, they don't necessarily).
    # Actually, the set of unmatched vertices in a *maximum* matching is not necessarily an independent set.
    # However, a greedy matching tends to leave "hard" vertices unmatched.
    # Let's just assign high score to vertices that are "early" in a greedy matching removal.
    # Alternative: Assign score based on "core number" from k-core decomposition.
    # Low core number = peripheral.
    
    core_numbers = np.zeros(n, dtype=np.float64)
    remaining = set(range(n))
    deg_rem = degrees.copy()
    
    # K-core decomposition
    while remaining:
        min_deg = min(deg_rem[v] for v in remaining)
        to_remove = [v for v in remaining if deg_rem[v] == min_deg]
        for v in to_remove:
            core_numbers[v] = min_deg
            remaining.remove(v)
            for u in neighbor_indices[v]:
                if u in remaining:
                    deg_rem[u] -= 1
    
    score_core = -core_numbers

    # Heuristic 3: Original Degree (Low degree first)
    score_deg = -degrees

    # Combine base scores
    # Normalize scores to [0, 1] or similar to allow mixing
    def normalize(s):
        min_s = np.min(s)
        max_s = np.max(s)
        if max_s - min_s < 1e-9:
            return np.zeros_like(s)
        return (s - min_s) / (max_s - min_s)

    base_score = 0.4 * normalize(score_periph) + 0.3 * normalize(score_core) + 0.3 * normalize(score_deg)
    
    # Add small random noise to break ties and allow SA to explore
    base_score = base_score + np.random.uniform(0, 0.1, n)
    
    current_scores = base_score
    current_size = greedy_size_from_scores(current_scores)
    best_scores = current_scores.copy()
    best_size = current_size
    
    # --- Step 2: Simulated Annealing on Score Vector ---
    # We want to maximize current_size.
    # Move: Perturb scores of a few random vertices.
    # This changes the order, potentially allowing the greedy decoder to pick a different set.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    max_iter = 5000 # Limit iterations to fit in time
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb
        # Pick 1-3 random vertices and shift their scores
        num_perturb = random.randint(1, 3)
        perturbed_scores = current_scores.copy()
        vertices_to_perturb = random.sample(range(n), num_perturb)
        for v in vertices_to_perturb:
            perturbed_scores[v] += np.random.normal(0, 0.5)
        
        new_size = greedy_size_from_scores(perturbed_scores)
        
        # Acceptance criterion
        delta = new_size - current_size
        if delta > 0 or random.random() < np.exp(delta / T):
            current_scores = perturbed_scores
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_scores = current_scores.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
