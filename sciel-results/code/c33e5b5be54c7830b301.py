import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    heuristic = np.ones((n, n)) * 1e-9
    
    # Precompute the total cost of moving to j from i and returning to 0
    # cost[i, j] = d[i, j] + d[j, 0]
    # Note: distance_matrix is symmetric, but we use d[j, 0] for return
    # d[i, j] is distance_matrix[i, j]
    # d[j, 0] is distance_matrix[j, 0]
    
    # Vectorized calculation of costs
    # d_i_j is distance_matrix
    # d_j_0 is distance_matrix[:, 0] reshaped to (n, 1)
    d_j_0 = distance_matrix[:, 0].reshape(-1, 1)
    costs = distance_matrix + d_j_0
    
    # Valid moves: j != i, j != 0, and cost[i, j] <= max_len
    # Also prize[0] is 0, so moving to 0 is not useful (and usually disallowed or low value)
    # The problem states "moves to an unvisited node j", and usually 0 is the start/end.
    # The loop in the original code skipped j==0. We should too.
    
    valid = np.zeros((n, n), dtype=bool)
    # Exclude diagonal (i == j) and column 0 (j == 0)
    np.fill_diagonal(valid, False)
    valid[:, 0] = False
    # Exclude moves that exceed max_len
    valid = valid & (costs <= max_len)
    
    # Calculate Score = Prize[j] / cost[i, j]
    # To avoid division by zero or invalid entries, we only calculate for valid entries
    # prizes is shape (n,)
    # We want Score[i, j] = prizes[j] / costs[i, j]
    
    scores = np.zeros((n, n))
    # Divide only where valid
    # costs > 0 is guaranteed for valid moves since distances are positive and max_len >= 3 > 0
    # But we must be careful with indexing.
    # prizes[j] needs to be broadcasted to (n, n) as (1, n)
    p = prizes.reshape(1, -1)
    
    # Initialize scores to 0
    scores = np.zeros((n, n))
    # Use np.where to avoid division by zero in invalid spots (though costs might be 0 if i=j, but we masked that)
    # Actually, costs[i,i] = d[i,i] + d[i,0] = 0 + d[i,0] > 0.
    # But we want scores to be 0 where not valid.
    
    # Calculate raw scores
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_scores = p / costs
    raw_scores[~valid] = 0.0
    
    # For each row i, find the maximum score among valid j
    # max_scores[i] = max_j raw_scores[i, j]
    max_scores = np.max(raw_scores, axis=1)
    
    # Calculate heuristic: H[i, j] = raw_scores[i, j] / (max_scores[i] + eps)
    # This is the "regret" style: how good is j compared to the best available option from i?
    
    eps = 1e-9
    # Reshape max_scores to (n, 1) for broadcasting
    max_scores_col = max_scores.reshape(-1, 1)
    
    # Avoid division by zero if max_scores is 0 (e.g. no valid moves, though max_len=3 with n=50 in unit square should always have moves)
    denom = max_scores_col + eps
    
    heuristic = raw_scores / denom
    
    # Ensure invalid entries (where valid is False) are 1e-9
    # And any non-finite or <= 0 values are 1e-9
    heuristic[~valid] = 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
