import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Total nodes N is not explicitly passed, but we can infer or assume 50.
    # However, the function doesn't know N.
    # We can estimate N based on the initial call? No, stateless.
    # But the problem says "50-city instances".
    # So we can hardcode N=50 or use a heuristic for lambda.
    
    # Let's use a lambda that depends on n_unvisited.
    # If n_unvisited is large, lambda is small.
    # If n_unvisited is small, lambda is large.
    
    # A simple curve: lambda = 1.0 / n_unvisited ?
    # Or lambda = (50 - n_unvisited) / 50.0
    
    # Let's try lambda = 0.5 * (1.0 - n_unvisited / 50.0)
    # This makes lambda 0 at start, 0.5 at end.
    
    # Actually, let's look at the cost structure.
    # The 2-step lookahead is $d_{curr} + d_{next}$.
    # The destination term is $d_{dest}$.
    
    # Let's just use a fixed small lambda, e.g., 0.2?
    # Or better, let's use the 2-step lookahead as the primary score.
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(d_sub, np.inf)
    d_next = np.min(d_sub, axis=1)
    
    # Base score: 2-step lookahead
    scores = d_curr + d_next
    
    # Add destination bias
    # If n is small, the bias should be stronger.
    # Let's use a weight that increases as n decreases.
    # weight = 1.0 / n_unvisited ?
    # If n=50, weight=0.02.
    # If n=2, weight=0.5.
    
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Scale the destination distance to be comparable to the lookahead score?
    # The lookahead score is a sum of 2 distances.
    # d_dest is 1 distance.
    # So maybe weight * d_dest.
    
    # Let's try weight = 0.5 / n_unvisited
    weights = 0.5 / n_unvisited
    scores += weights * d_dest
    
    min_idx = np.argmin(scores)
    return int(unvisited_nodes[min_idx])
