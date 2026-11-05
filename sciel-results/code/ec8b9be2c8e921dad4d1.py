import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores via Markov Chain stationary distribution analysis.
    
    Strategy:
    - Construct a transition matrix P where P[i, j] represents the probability
      of transitioning from vertex i to vertex j in a random greedy process.
      We approximate this by assuming that from any vertex i, the next "active"
      vertex is chosen uniformly at random from the remaining non-neighbor vertices,
      or more simply, we model a random walk on the graph where the transition
      probabilities are inversely proportional to degree, favoring movement to
      low-degree nodes.
    - However, a more direct interpretation for MIS greedy decoding:
      The greedy decoder picks vertices in descending score order.
      If we view the selection process as a random walk that "blocks" neighbors,
      vertices that are rarely visited in a stationary random walk on the graph
      are likely to be in the periphery and less likely to be blocked by early
      selections in the core.
    - We compute the stationary distribution pi of a random walk on the graph.
      Vertices with low pi (low stationary probability) are considered "peripheral"
      and are given higher priority scores, as they are less likely to be adjacent
      to the high-traffic core nodes that get picked early and block their neighbors.
    - We combine this with a degree-based heuristic: low degree is generally good.
    - Finally, we perform a local search refinement on the resulting order to maximize
      the actual independent set size, using the Markov-informed order as the start.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # Precompute neighbor lists and degrees
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    deg = adj.sum(axis=1).astype(np.float64)
    
    # Avoid division by zero for isolated nodes
    deg_safe = np.maximum(deg, 1.0)

    # Step 1: Construct Transition Matrix for Random Walk
    # P[i, j] = 1 / deg(i) if (i, j) is an edge, else 0
    # For isolated nodes, we treat them as self-loops or handle separately.
    # Since we want stationary distribution, we need a stochastic matrix.
    # Let's use a lazy random walk to ensure aperiodicity, or just standard.
    # P = D^{-1} A
    # But if deg[i] == 0, P[i, i] = 1? Or we can just set P[i, j] = 0 for all j and handle later.
    # Actually, for the purpose of "peripherality", isolated nodes are the most peripheral.
    
    # Let's construct the transition matrix W where W[i, j] = 1 if edge exists.
    # Then normalize rows.
    W = adj.astype(np.float64)
    
    # Handle isolated nodes: if deg[i] == 0, set W[i, i] = 1 to make it stochastic?
    # Or just leave row sum 0. Power iteration might fail.
    # Let's add a small self-loop to all nodes to ensure connectivity/stochasticity for the walk model.
    # This models a "lazy" walk or just ensures the matrix is well-behaved.
    W += np.eye(n) * 0.01  # Small self-loop to avoid zero rows and ensure aperiodicity
    
    row_sums = W.sum(axis=1)
    # Normalize rows to sum to 1
    P = W / row_sums[:, np.newaxis]
    
    # Step 2: Compute Stationary Distribution via Power Iteration
    # pi_{k+1} = pi_k * P
    # Initialize pi uniformly
    pi = np.ones(n) / n
    
    # Power iteration
    # Convergence is typically fast for small graphs like n=200
    max_iter = 1000
    tol = 1e-8
    for _ in range(max_iter):
        if time.time() - start_time > deadline - 30:
            break
        new_pi = pi @ P
        # Check convergence
        if np.linalg.norm(new_pi - pi, 1) < tol:
            pi = new_pi
            break
        pi = new_pi
    
    # Step 3: Define Initial Scores
    # Hypothesis: Vertices with LOW stationary probability (pi) are in the periphery.
    # In a greedy MIS decoder, picking peripheral nodes first might be better because
    # they don't block as many core nodes, or rather, core nodes block many, so if we pick
    # peripheral nodes, we leave the core intact? No, that's not right.
    # If we pick a core node, it blocks many neighbors.
    # If we pick a peripheral node, it blocks few neighbors.
    # So picking peripheral nodes is "safe" but might not maximize count if the core is dense.
    # Actually, standard MIS heuristics often pick low-degree nodes first.
    # Low pi often correlates with low degree or being on the fringe.
    
    # Let's try: Score = -pi (lower pi -> higher score)
    # But also consider degree.
    # Let's combine: Score = (1 / (1 + deg)) * (1 / (1 + pi * n))
    # Or simply: Score = -pi - deg/n
    
    # Let's test the pure "peripheral" hypothesis first:
    # Score is inversely proportional to pi.
    # To avoid division by zero, add epsilon.
    scores = 1.0 / (pi + 1e-6)
    
    # Normalize scores to [0, 1] for stability, though order is what matters
    scores_min = scores.min()
    scores_max = scores.max()
    if scores_max > scores_min:
        scores = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores = np.ones(n) / n

    # Step 4: Local Search Refinement
    # Convert scores to an order
    order = np.argsort(-scores, kind='stable')
    
    def greedy_is_size(order):
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count

    current_size = greedy_is_size(order)
    best_order = order.copy()
    best_size = current_size
    
    # Hill Climbing with Random Swaps
    rng = np.random.default_rng(42)
    
    # Time budget for local search
    remaining_time = deadline - 10 - (time.time() - start_time)
    if remaining_time > 1:
        # We will perform random swaps and accept if improvement
        # To speed up, we can batch checks or just do many iterations
        n_attempts = 5000  # Adjust based on time
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            for _ in range(n_attempts):
                if time.time() - start_time > deadline - 10:
                    break
                
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                
                v_i = best_order[i]
                v_j = best_order[j]
                
                # Swap
                best_order[i], best_order[j] = best_order[j], best_order[i]
                new_size = greedy_is_size(best_order)
                
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                else:
                    # Swap back
                    best_order[i], best_order[j] = best_order[j], best_order[i]
    
    # Step 5: Convert best_order to final scores
    # Vertex at position k gets score n - k
    final_scores = np.zeros(n)
    for k, v in enumerate(best_order):
        final_scores[v] = n - k
        
    return final_scores
