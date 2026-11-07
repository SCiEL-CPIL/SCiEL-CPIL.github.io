"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix using a regret-based opportunity cost model.
    
    The heuristic biases ants towards moves that preserve access to high-value nodes
    by penalizing moves that would make high-prize nodes unreachable.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base efficiency: prize per unit distance (including return cost)
    # d_ij + d_j0 is the additional cost of visiting j and returning
    d_j0 = distance_matrix[:, 0]  # cost from each node back to 0
    eps = 1e-9
    
    # Full detour cost for edge i->j: d[i,j] + d[j,0]
    full_cost = distance_matrix + d_j0[None, :]  # shape (n, n)
    
    # Base heuristic: prize_j / full_cost_ij
    base_eta = (prizes[None, :] + eps) / (full_cost + eps)
    
    # Now compute the regret/opportunity cost component
    # For each edge i->j, look at all other nodes k that are "reachable" from i
    # given that we've already committed to going to j (or rather, the remaining budget)
    # 
    # Actually, let's think about this differently:
    # The key insight is that if we go i->j, we spend d[i,j] of our budget.
    # The remaining budget for further moves is max_len - current_length - d[i,j] - d[j,0]
    # But since we don't know current_length in the static heuristic, we use a proxy.
    #
    # Better approach: For each pair (i, j), compute how many high-value nodes become
    # "harder to reach" if we take this edge. Specifically, for node k to be visitable
    # after i->j, we need d[i,j] + d[j,k] + d[k,0] <= max_len (roughly, ignoring prior path).
    # 
    # Let's define: for edge i->j, the opportunity cost is the sum of prizes of nodes k
    # that are reachable from i directly (d[i,k] + d[k,0] <= max_len) but NOT reachable
    # via j (d[i,j] + d[j,k] + d[k,0] > max_len).
    #
    # This is O(n^3) which for n=50 is 125,000 operations - very fast.
    
    # Precompute reachability masks
    # reach_direct[i, k] = True if d[i,k] + d[k,0] <= max_len
    reach_direct = (distance_matrix + d_j0[None, :]) <= max_len
    
    # For each edge (i, j), compute which nodes k are reachable via j
    # reach_via_j[i, j, k] = True if d[i,j] + d[j,k] + d[k,0] <= max_len
    # This is a 3D tensor of size n x n x n = 125,000 booleans for n=50
    
    # Vectorized computation
    # cost_via_j[i, j, k] = d[i,j] + d[j,k] + d[k,0]
    d_ij = distance_matrix[:, :, None]  # (n, n, 1)
    d_jk = distance_matrix[:, None, :]  # (n, 1, n)
    d_k0 = d_j0[None, None, :]  # (1, 1, n)
    
    cost_via_j = d_ij + d_jk + d_k0  # (n, n, n)
    reach_via_j = cost_via_j <= max_len  # (n, n, n)
    
    # Opportunity cost for edge (i, j): sum of prizes[k] where reach_direct[i, k] 
    # but NOT reach_via_j[i, j, k], and k != 0 (node 0 has prize 0 anyway)
    # We want to penalize edges that block access to high-value nodes
    
    # blocked[i, j, k] = reach_direct[i, k] AND NOT reach_via_j[i, j, k]
    blocked = reach_direct[:, :, None] & (~reach_via_j)  # (n, n, n)
    
    # Opportunity cost: sum of prizes of blocked nodes
    opp_cost = np.sum(blocked * prizes[None, None, :], axis=2)  # (n, n)
    
    # The heuristic should be high when:
    # 1. The edge is efficient (high prize per cost)
    # 2. The edge doesn't block access to other high-value nodes
    #
    # Combine: eta[i, j] = base_eta[i, j] / (1 + opp_cost[i, j])^alpha
    # where alpha controls the strength of the regret penalty
    
    alpha = 1.5
    regret_penalty = (1.0 + opp_cost) ** alpha
    
    eta = base_eta / regret_penalty
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(eta, 0.0)
    
    # Ensure all values are positive (replace non-positive with 1e-9)
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return eta
    
    # Normalize to avoid extreme values
    # Scale so that the median is around 1
    positive_vals = eta[eta > 1e-8]
    if len(positive_vals) > 0:
        median_val = np.median(positive_vals)
        if median_val > 0:
            eta = eta / median_val
    
    # Final safety: ensure finite values
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
