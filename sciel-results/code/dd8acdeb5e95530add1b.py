# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy: Directly optimize the vertex ordering (permutation) using Simulated
    Annealing with incremental delta-evaluation. We start from a degree-based
    ordering and iteratively improve it by moving vertices to better positions,
    evaluating the change in greedy MIS size efficiently.
    """
    deadline = time.time() + 115  # Safety margin before 120s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.array([0.0])

    # Precompute neighbor sets as boolean arrays for fast checks
    # neighbors[v] is a boolean array of length n
    neighbors = adjacency  # Already boolean matrix
    
    def evaluate_order(order):
        """
        Evaluate the greedy independent set size for a given order.
        Returns (count, picked_mask).
        """
        picked = np.zeros(n, dtype=bool)
        forbidden = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if forbidden[v]:
                continue
            picked[v] = True
            count += 1
            # Mark neighbors as forbidden
            # Use vectorized operation for speed
            forbidden[neighbors[v]] = True
        return count, picked

    def delta_eval(order, i, j):
        """
        Compute the change in greedy MIS size if we move the vertex at position i to position j
        (shifting elements in between).
        This is an approximation or exact delta depending on implementation.
        For simplicity and correctness, we will do a local re-evaluation of the affected segment
        or just full evaluation if n is small enough, but for n=200, full eval is ~200*200 = 40k ops,
        which is fast in numpy/python if optimized.
        However, SA needs many iterations. 
        Let's try a different move: Swap two vertices in the order.
        Delta for swap is hard to compute exactly without re-running.
        
        Alternative: Use a greedy construction that is reversible? No.
        
        Let's stick to full evaluation but optimize the loop.
        n=200. Full eval is 200 iterations. Each iteration does a boolean array update.
        200 * (200 ops) = 40k ops. In Python, this is ~1ms? Maybe 5ms.
        120s / 5ms = 24,000 iterations. This is a decent number of SA steps.
        
        Let's implement a fast evaluator.
        """
        picked = np.zeros(n, dtype=bool)
        forbidden = np.zeros(n, dtype=bool)
        count = 0
        # Local variables for speed
        nbrs = neighbors
        for v in order:
            if not forbidden[v]:
                picked[v] = True
                count += 1
                forbidden[nbrs[v]] = True
        return count

    # Initial ordering: sort by degree (ascending) - low degree first
    degree = adjacency.sum(axis=1)
    initial_order = np.argsort(degree, kind='stable')
    current_order = initial_order.copy()
    current_score = evaluate_order(current_order)[0]
    
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995  # Cooling rate
    T = T_start
    
    # We want to maximize the score (independent set size).
    # Move: Swap two random positions in the order.
    
    elapsed_limit = deadline
    rng = random.Random(42)
    
    # To make it faster, we can precompute the neighbor lists as lists of ints?
    # No, boolean arrays are fine for numpy vectorization of the update, but the loop is python.
    # Let's try to speed up the evaluator.
    
    def fast_eval(order):
        forbidden = np.zeros(n, dtype=bool)
        count = 0
        nbrs = neighbors
        for v in order:
            if not forbidden[v]:
                count += 1
                forbidden[nbrs[v]] = True
        return count

    current_score = fast_eval(current_order)
    best_score = current_score
    best_order = current_order.copy()
    
    # Precompute neighbor lists as lists for potentially faster iteration if we change strategy?
    # No, the bottleneck is the python loop.
    
    # Let's try to optimize the inner loop of fast_eval.
    # Using a set of forbidden nodes?
    # Forbidden is a boolean array. Checking `if not forbidden[v]` is O(1).
    # Updating `forbidden[nbrs[v]] = True` is O(deg(v)).
    # Total cost per eval is O(n + sum(deg(v) for v in picked)).
    # Worst case O(n^2). Average O(n * avg_deg).
    # For ER graph, avg_deg ~ n*p.
    
    # Let's run SA
    steps = 0
    while time.time() < deadline:
        # Pick two random positions to swap
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        # Perform swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_score = fast_eval(current_order)
        
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Accept move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = current_order.copy()
        else:
            # Reject move, revert swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # Optional: restart with higher T if stuck? 
            # Or just keep going with minimal T to explore local minima slightly.
            
        steps += 1
        
        # Safety check every 1000 steps
        if steps % 1000 == 0 and time.time() > deadline:
            break

    # Convert best_order back to scores
    # The decoder sorts by descending score.
    # So vertex at index 0 in best_order should have the highest score.
    # Vertex at index n-1 in best_order should have the lowest score.
    scores = np.zeros(n)
    # Assign scores such that argsort(-scores) == best_order
    # Let score[v] = n - 1 - position_of_v_in_best_order
    # Then higher score -> lower position index -> picked earlier.
    for idx, v in enumerate(best_order):
        scores[v] = n - 1 - idx
        
    return scores
# EVOLVE-BLOCK-END
