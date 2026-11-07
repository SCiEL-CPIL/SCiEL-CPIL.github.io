"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    d = distance_matrix
    
    # Compute distance from each node to node 0 (start)
    d_to_0 = d[:, 0]
    
    # For each node j, compute the "cost" of visiting it:
    # The minimum additional distance needed to visit j from any i and return to 0
    # This is d[i,j] + d[j,0] for the transition i->j, but since we're computing
    # a static heuristic, we use d[j,0] as a proxy for the "return cost"
    
    # Key insight: nodes that are close to node 0 are safer to visit because
    # they leave more budget for other visits. Also, high prize nodes are attractive.
    
    # Compute a "safety" score: how much budget remains after visiting j
    # If we visit j, we need at least d[j,0] to get back (plus some slack)
    # So nodes with small d[j,0] are safer
    
    # Heuristic design:
    # 1. Prize attraction: higher prize = more attractive
    # 2. Distance penalty: shorter distance from current node = more attractive  
    # 3. Safety: nodes closer to start are safer (leave more budget)
    
    # Since the heuristic is static and computed once, we design it to:
    # - Favor high prize nodes
    # - Penalize nodes that are far from the start (risky)
    # - The transition cost d[i,j] is handled by the denominator
    
    # Let's use a combination:
    # heuristic[i,j] = (prize[j] + eps) / (d[i,j] + eps) * safety_factor[j]
    
    # Where safety_factor[j] rewards nodes that are close to start
    # This encourages ants to visit "safe" high-value nodes first
    
    # Safety factor: exp(-alpha * d_to_0[j] / max_len) or similar
    # Nodes close to start get higher safety factor
    
    # Let's try: safety_factor = 1 / (1 + d_to_0[j])
    # This gives higher weight to nodes closer to start
    
    d_to_0_safe = 1e-9 + d_to_0
    safety_factor = 1.0 / d_to_0_safe
    
    # Also consider: nodes that are far from start might still be worth visiting
    # if they have very high prizes. So let's use a softer penalty.
    
    # Alternative: use a power law
    # safety_factor = (max_len / (max_len + d_to_0[j]))^beta
    
    # Let's try multiple approaches and pick a good one
    # Since we can't test, let's use a reasonable default
    
    # Approach: 
    # heuristic[i,j] = (prize[j] + 1e-9) / (d[i,j] + 1e-9) * (1 / (1 + d_to_0[j]))
    
    # This combines:
    # - Prize in numerator
    # - Transition distance in denominator  
    # - Safety factor (closer to start = better)
    
    prizes_col = prizes[None, :] + 1e-9
    d_safe = d + 1e-9
    safety_row = (1.0 / (1.0 + d_to_0))[None, :]
    
    heuristic = (prizes_col / d_safe) * safety_row
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
