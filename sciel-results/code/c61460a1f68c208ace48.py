"""Constructive TSP heuristic: Lookahead NN with destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = None
_DEADLINE = 150.0
_SAFETY = 5.0

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    
    if n == 0:
        return int(current_node)
    
    if n == 1:
        return int(unvisited_nodes[0])

    # Hard deadline guard
    if _elapsed() > _DEADLINE - _SAFETY:
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])

    try:
        # 1. Immediate distances from current node to all unvisited
        d_cur = distance_matrix[current_node, unvisited_nodes]
        
        # 2. Distances from all unvisited to destination
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # 3. Estimate future local cost: min distance from each candidate to other unvisited nodes
        # This is O(N^2) but N is small (<=50).
        # D_sub is the submatrix of distance_matrix for unvisited nodes
        D_sub = distance_matrix[unvisited_nodes, unvisited_nodes]
        
        # Set diagonal to infinity so we don't pick the node itself as "next"
        np.fill_diagonal(D_sub, np.inf)
        
        # For each candidate j, the cost of the *next* step is the min distance to any other unvisited node
        d_min_next = np.min(D_sub, axis=1)
        
        # If n==2, d_min_next will be the distance between the two nodes, which is correct.
        # If n>2, it's the nearest neighbor among unvisited.
        
        # 4. Phase-aware weighting
        # Total nodes is typically 50. We estimate progress based on remaining nodes.
        # Assuming standard 50-city instances. If n is very small, progress is high.
        # Let's assume total cities N=50 for normalization, or just use n.
        # If we have 50 cities, start with n=50, end with n=1.
        # Progress p goes from 0 to 1.
        # Using n directly: p = (50 - n) / 49. 
        # To be robust to different N, we can just use n.
        # Early: n large -> p small -> mu small.
        # Late: n small -> p large -> mu large.
        
        # Heuristic: 
        # - d_cur is the primary driver.
        # - d_min_next penalizes moving to a node that is isolated (far from other options).
        # - d_dest pulls us towards the destination to close the loop.
        
        # Weights:
        # lambda: weight for lookahead local cost. 0.5 seems reasonable.
        # mu: weight for destination. Should be 0 early, 1 late.
        
        # Let's use a sigmoid or polynomial for mu.
        # If n > 20, mu is small. If n < 10, mu is large.
        # p = 1.0 - (n / 50.0) # Rough estimate of progress
        # Let's just use n to determine mu.
        # If n is close to total, mu=0. If n is 1, mu=1.
        # Since we don't know total N exactly (though it's 50), let's assume N=50.
        # If the instance size varies, this might be slightly off, but 50 is specified in task.
        N_total = 50
        progress = (N_total - n) / max(N_total - 1, 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        
        # Mu increases quadratically to keep destination bias low until late
        mu = progress ** 2
        lambda_ = 0.5
        
        # Calculate score
        # Score = d_cur + lambda * d_min_next + mu * d_dest
        # We need to ensure scales are comparable. d_cur, d_min_next, d_dest are all Euclidean distances.
        # So direct addition is valid.
        
        score = d_cur + lambda_ * d_min_next + mu * d_dest
        
        best_idx = int(np.argmin(score))
        
        return int(unvisited_nodes[best_idx])
        
    except Exception:
        # Fallback to simple NN
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
