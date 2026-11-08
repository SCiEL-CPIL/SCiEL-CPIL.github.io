"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]

    # Mean degree of neighbors
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]

    # Triangle count per vertex: number of edges in the neighborhood of i
    # This is a measure of how "clique-like" the neighborhood is.
    # If a vertex is in many triangles, its neighbors are well-connected, 
    # so picking it blocks a dense cluster, which might be good or bad depending on context.
    # Actually, if neighbors are well-connected, they block each other anyway.
    # So picking a vertex with high triangle count might be less "wasteful" because its neighbors are already blocking each other.
    # Let's compute triangle count.
    # A^2[i,j] is the number of common neighbors.
    # Triangles involving i = 0.5 * sum_{j in N(i)} (A^2[i,j])
    # But A^2[i,j] for j in N(i) is the number of common neighbors of i and j.
    # Each triangle (i,j,k) is counted twice in the sum for i (once for j, once for k).
    # So triangles_i = 0.5 * sum_{j in N(i)} A^2[i,j]
    
    # Computing full A^2 is O(n^3) or O(n^2) with matrix mult. For n=200, it's fast.
    A_sq = adjacency @ adjacency
    triangle_count = np.zeros(n)
    for i in range(n):
        if degrees[i] > 0:
            triangle_count[i] = 0.5 * A_sq[i, neighbor_lists[i]].sum()

    def simulate_greedy(scores):
        """Simulate greedy decoder and return size of independent set."""
        order = np.argsort(-scores)  # descending score
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                selected[neighbor_lists[v]] = True
        return count

    # Generate candidate score vectors
    candidates = []
    
    # 1. Seed: -degree
    candidates.append(-degrees)
    
    # 2. -degree + avg_neighbor_degree
    candidates.append(-degrees + avg_neighbor_degree)
    
    # 3. -degree / (1 + avg_neighbor_degree)
    candidates.append(-degrees / (1 + avg_neighbor_degree))
    
    # 4. -degree * (1 + avg_neighbor_degree)
    candidates.append(-degrees * (1 + avg_neighbor_degree))
    
    # 5. Random perturbation of degree to break ties
    rng = np.random.RandomState(42)
    candidates.append(-degrees + rng.normal(0, 0.1, n))
    
    # 6. -degree + 2 * avg_neighbor_degree
    candidates.append(-degrees + 2 * avg_neighbor_degree)
    
    # 7. -degree * avg_neighbor_degree
    candidates.append(-degrees * avg_neighbor_degree)
    
    # 8. -degree + 0.5 * avg_neighbor_degree
    candidates.append(-degrees + 0.5 * avg_neighbor_degree)
    
    # 9. -degree + triangle_count
    candidates.append(-degrees + triangle_count)
    
    # 10. -degree + 0.5 * triangle_count
    candidates.append(-degrees + 0.5 * triangle_count)
    
    # 11. -degree - triangle_count (prefer low triangle count)
    candidates.append(-degrees - triangle_count)
    
    # 12. -degree / (1 + triangle_count)
    candidates.append(-degrees / (1 + triangle_count))
    
    # 13. -degree + avg_neighbor_degree + triangle_count
    candidates.append(-degrees + avg_neighbor_degree + triangle_count)
    
    # 14. -degree + 0.5 * avg_neighbor_degree + 0.5 * triangle_count
    candidates.append(-degrees + 0.5 * avg_neighbor_degree + 0.5 * triangle_count)
    
    # 15. -degree + 0.2 * avg_neighbor_degree + 0.8 * triangle_count
    candidates.append(-degrees + 0.2 * avg_neighbor_degree + 0.8 * triangle_count)
    
    # Evaluate each candidate
    best_score = -np.inf
    best_candidate = candidates[0]

    for cand in candidates:
        if time.time() > deadline - 5:
            break
        sim_size = simulate_greedy(cand)
        if sim_size > best_score:
            best_score = sim_size
            best_candidate = cand

    # If we have time, try to optimize further with local search or more heuristics
    if time.time() < deadline - 10:
        # Try a grid search over weights for avg_neighbor_degree and triangle_count
        weights_an = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        weights_tc = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        
        for w_an in weights_an:
            if time.time() > deadline - 5:
                break
            for w_tc in weights_tc:
                if time.time() > deadline - 5:
                    break
                cand = -degrees + w_an * avg_neighbor_degree + w_tc * triangle_count
                sim_size = simulate_greedy(cand)
                if sim_size > best_score:
                    best_score = sim_size
                    best_candidate = cand

    return best_candidate
# EVOLVE-BLOCK-END
