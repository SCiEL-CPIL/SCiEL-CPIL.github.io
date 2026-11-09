"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = (adj > 0.5).astype(np.int8)
    degrees = adj_bool.sum(axis=1).astype(np.int32)
    
    # Precompute neighbor lists
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_scores(scores):
        """Compute the size of the greedy independent set given a score vector."""
        order = np.argsort(-scores, kind='stable')
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    # --- Core Strategy: Min-Degree Peeling (Deterministic Combinatorial Approach) ---
    # The greedy decoder picks vertices in descending score order, skipping neighbors of picked vertices.
    # This is equivalent to: repeatedly pick the "best" vertex from the remaining graph, remove it and its neighbors.
    # If we assign scores such that the highest-score vertex is always the one the peeling algorithm picks first,
    # we can directly construct an ordering that maximizes the greedy MIS size.
    #
    # Min-degree peeling: at each step, pick the vertex with minimum degree in the current remaining graph.
    # This is a well-known heuristic for MIS and tends to produce large independent sets.
    
    def min_degree_peeling_order(adj_bool, degrees, neighbor_indices, n):
        """
        Perform min-degree peeling and return the order in which vertices are selected.
        The first vertex in the order gets the highest score.
        """
        # Use a heap for efficient min-degree selection
        # Each entry: (degree, vertex)
        active = set(range(n))
        deg = degrees.copy()
        heap = [(deg[v], v) for v in range(n)]
        heapq.heapify(heap)
        
        # Track which vertices have been processed (selected or removed as neighbors)
        processed = np.zeros(n, dtype=bool)
        
        order = []
        for _ in range(n):
            if time.time() > deadline - 10.0:
                break
                
            # Pop the vertex with minimum degree from the heap
            # Skip stale entries
            while heap:
                d, v = heapq.heappop(heap)
                if v in active and not processed[v] and deg[v] == d:
                    break
            else:
                # Heap exhausted, pick any active vertex
                if active:
                    v = min(active)
                    d = deg[v]
                else:
                    break
            
            if v not in active or processed[v]:
                continue
            
            # Select vertex v
            order.append(v)
            processed[v] = True
            active.discard(v)
            
            # Remove neighbors of v from active
            for u in neighbor_indices[v]:
                if u in active and not processed[u]:
                    active.discard(u)
                    processed[u] = True  # Mark as removed (not selected, but blocked)
                    # Update degrees of remaining active vertices
                    for w in neighbor_indices[u]:
                        if w in active and not processed[w]:
                            deg[w] -= 1
                            # Push updated degree into heap
                            heapq.heappush(heap, (deg[w], w))
        
        return order

    # --- Alternative: Randomized Min-Degree Peeling with Restart ---
    # To get diversity and potentially better results, try multiple random perturbations
    # of the initial degrees to break ties differently.
    
    def randomized_min_degree_peeling(adj_bool, degrees, neighbor_indices, n, seed):
        """
        Min-degree peeling with random tie-breaking based on a seed.
        """
        rng = np.random.RandomState(seed)
        # Add small random noise to break ties in degree
        perturbed_deg = degrees.astype(np.float64) + rng.uniform(0, 0.5, n)
        
        active = set(range(n))
        deg = perturbed_deg.copy()
        # Use a list-based approach for simplicity, or a heap with lazy deletion
        # For n=200, a simple O(n^2) approach is fast enough
        
        order = []
        for _ in range(n):
            if time.time() > deadline - 10.0:
                break
                
            if not active:
                break
            
            # Find vertex with minimum perturbed degree
            min_deg = float('inf')
            best_v = -1
            for v in active:
                if deg[v] < min_deg:
                    min_deg = deg[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            # Select vertex best_v
            order.append(best_v)
            active.discard(best_v)
            
            # Remove neighbors
            for u in neighbor_indices[best_v]:
                if u in active:
                    active.discard(u)
                    # Update degrees of remaining active vertices
                    for w in neighbor_indices[u]:
                        if w in active:
                            deg[w] -= 1
        
        return order

    # --- Construct Scores from Peeling Order ---
    # The vertex selected first in peeling gets the highest score.
    # We assign scores as: score[v] = n - position_in_order[v]
    # This ensures the greedy decoder picks vertices in the same order as the peeling.
    
    def order_to_scores(order, n):
        """Convert a peeling order to scores such that argsort(-scores) matches the order."""
        scores = np.zeros(n, dtype=np.float64)
        for i, v in enumerate(order):
            scores[v] = float(n - i)  # First selected gets highest score
        # Vertices not in order (if peeling terminated early) get low scores
        for v in range(n):
            if scores[v] == 0.0:
                scores[v] = 0.5
        return scores

    # --- Multi-Restart: Try Different Strategies ---
    
    best_scores = None
    best_size = -1
    
    # Strategy 1: Deterministic min-degree peeling
    order1 = min_degree_peeling_order(adj_bool, degrees, neighbor_indices, n)
    scores1 = order_to_scores(order1, n)
    size1 = greedy_size_from_scores(scores1)
    
    if best_size < size1:
        best_size = size1
        best_scores = scores1
    
    # Strategy 2: Multiple randomized min-degree peelings
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        seed = 42 + r
        order_r = randomized_min_degree_peeling(adj_bool, degrees, neighbor_indices, n, seed)
        scores_r = order_to_scores(order_r, n)
        size_r = greedy_size_from_scores(scores_r)
        
        if size_r > best_size:
            best_size = size_r
            best_scores = scores_r
    
    # Strategy 3: Hybrid - combine peeling order with structural features
    # Use core number and distance-2 degree to create a composite score
    # that biases the peeling toward structurally "good" vertices.
    
    # Compute core numbers
    core_numbers = np.zeros(n, dtype=np.float64)
    remaining = set(range(n))
    deg_rem = degrees.astype(np.float64).copy()
    
    while remaining:
        min_deg = min(deg_rem[v] for v in remaining)
        to_remove = [v for v in remaining if deg_rem[v] == min_deg]
        for v in to_remove:
            core_numbers[v] = min_deg
            remaining.discard(v)
            for u in neighbor_indices[v]:
                if u in remaining:
                    deg_rem[u] -= 1
    
    # Compute distance-2 degrees
    try:
        A2 = adj_bool @ adj_bool
        deg2 = A2.sum(axis=1) - degrees
    except Exception:
        deg2 = degrees
    
    def normalize(s):
        min_s = np.min(s)
        max_s = np.max(s)
        if max_s - min_s < 1e-9:
            return np.zeros_like(s)
        return (s - min_s) / (max_s - min_s)
    
    # Composite structural score: prefer low core, low deg2, low degree
    struct_score = 0.4 * normalize(-core_numbers) + 0.3 * normalize(-deg2) + 0.3 * normalize(-degrees)
    
    # Strategy 4: Peeling with structural bias
    # Modify the peeling to prefer vertices that are both low-degree AND structurally peripheral
    def biased_min_degree_peeling(adj_bool, degrees, neighbor_indices, n, struct_score, seed):
        rng = np.random.RandomState(seed)
        # Combine degree and structural score
        # Lower degree + higher structural score = better candidate
        composite = degrees.astype(np.float64) - 2.0 * struct_score  # Penalize high degree, reward peripheral
        composite += rng.uniform(0, 0.1, n)  # Small noise for tie-breaking
        
        active = set(range(n))
        comp = composite.copy()
        
        order = []
        for _ in range(n):
            if time.time() > deadline - 10.0:
                break
            
            if not active:
                break
            
            # Find vertex with minimum composite score
            min_comp = float('inf')
            best_v = -1
            for v in active:
                if comp[v] < min_comp:
                    min_comp = comp[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            order.append(best_v)
            active.discard(best_v)
            
            for u in neighbor_indices[best_v]:
                if u in active:
                    active.discard(u)
                    for w in neighbor_indices[u]:
                        if w in active:
                            comp[w] -= 1  # Degree decreases, making it more attractive
        
        return order
    
    # Try biased peeling with multiple seeds
    for r in range(10):
        if time.time() > deadline - 15.0:
            break
        
        seed = 1000 + r
        order_r = biased_min_degree_peeling(adj_bool, degrees, neighbor_indices, n, struct_score, seed)
        scores_r = order_to_scores(order_r, n)
        size_r = greedy_size_from_scores(scores_r)
        
        if size_r > best_size:
            best_size = size_r
            best_scores = scores_r
    
    # Final safety: if no valid scores, fall back to simple degree-based
    if best_scores is None:
        best_scores = -degrees.astype(np.float64)
    
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
