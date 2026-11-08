"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    # Ensure it's a numpy array of floats
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Calculate degrees
    degrees = adj.sum(axis=1)
    
    # Handle isolated vertices or zero degree cases to avoid division by zero
    # We want to propagate score from neighbors.
    # Standard PageRank-like update: score_new[i] = (1-d) + d * sum(score[j] / out_degree[j] for j in neighbors)
    # Since it's undirected, in_degree = out_degree = degree.
    # To ensure vertices with no neighbors don't disappear, we add a small base score or handle separately.
    # For MIS, we generally want vertices that are "central" in a way that allows many non-neighbors to be picked?
    # Actually, for Greedy MIS, we want to pick vertices that "block" the fewest other high-value vertices?
    # Or vertices that are "isolated" in the subgraph of remaining candidates?
    # 
    # Let's try a different angle: 
    # The greedy algorithm picks in order. If we pick a high-degree vertex early, we remove its neighbors.
    # If we pick a low-degree vertex early, we remove fewer neighbors.
    # However, the observed child p4 (score 1.026) used:
    # - Low degree (primary)
    # - High average neighbor degree (secondary? or combined?)
    # The diff showed:
    # avg_neighbor_degree = sum_neighbor_degrees / degrees
    # It likely returned something like -degree + alpha * avg_neighbor_degree or similar.
    # 
    # Let's try a global centrality measure that favors vertices in "loose" parts of the graph.
    # Betweenness centrality is expensive.
    # Eigenvector centrality favors vertices connected to high-degree vertices. 
    # For MIS, we might actually want the OPPOSITE of standard eigenvector centrality?
    # Or perhaps we want vertices that are part of large independent sets in their local neighborhood?
    #
    # Let's try an iterative relaxation that approximates the "independence" of a vertex.
    # A vertex is "good" if its neighbors are "bad" (i.e., likely to be skipped or low value).
    # This is a fixed-point iteration: score[i] = 1 / (1 + sum(score[neighbors]))?
    # Or simpler: score[i] = 1 / degree[i] is the seed.
    # 
    # Let's try: Score[i] = 1 / (1 + sum_{j in N(i)} Score[j])
    # This creates a recursive dependency. Vertices whose neighbors have low scores get high scores.
    # This tends to alternate high/low scores across the graph (bipartite-like).
    # In a bipartite graph, one side gets high scores, the other low. The greedy picker will pick all high-score vertices (one side), which is a Maximum Independent Set for bipartite graphs!
    # This seems like a very strong heuristic for MIS.
    
    # Initialize scores
    scores = np.ones(n)
    
    # Iterative refinement
    # We want to converge to a state where scores[i] * (1 + sum(scores[N(i)])) ~ 1
    # Let's use a damped update to ensure convergence and avoid oscillation.
    # new_scores[i] = 1 / (1 + sum(scores[N(i)]))
    # But if we update simultaneously, it might oscillate.
    # Let's use Jacobi iteration with damping.
    
    # Precompute neighbor sums efficiently? 
    # sum(scores[N(i)]) is adj @ scores.
    
    iterations = 0
    max_iterations = 10000
    tol = 1e-9
    
    # Ensure no division by zero issues in the denominator (1 + sum is always >= 1)
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
            
        # Calculate sum of neighbor scores
        neighbor_sums = adj @ scores
        
        # Target scores
        target_scores = 1.0 / (1.0 + neighbor_sums)
        
        # Damped update
        # We want to move towards target_scores.
        # If we just set scores = target_scores, it might oscillate for non-bipartite graphs.
        # Let's mix with previous scores.
        scores = 0.5 * scores + 0.5 * target_scores
        
        # Check convergence
        # if np.max(np.abs(scores - target_scores)) < tol:
        #     break
        # This check is expensive. Let's just run a fixed number of iterations or rely on time.
        # Given N=200, 1000 iterations is very fast (200x200 matrix mult is ~8M ops, 1000 is 8B ops, maybe 1-2 seconds).
        # We can afford many iterations.
        
        iterations += 1
        
        # Early exit if time is tight
        if time.time() > deadline - 2.0:
            break

    # The greedy decoder picks in descending score.
    # So we return the scores directly.
    # Vertices with high scores will be picked first.
    # This heuristic effectively identifies a "maximum" side of a near-bipartite structure or isolates vertices in sparse regions.
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
