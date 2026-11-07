"""
Strategy: Clarke-Wright Savings Heuristic
Instead of a local geometric heuristic (like inverse distance or angle), 
this strategy computes a heuristic matrix based on the Clarke-Wright Savings 
principle. The heuristic value for edge (i, j) is proportional to the savings 
achieved by visiting j immediately after i, compared to visiting them in 
separate trips from the depot.

Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)

This directly encodes the "merging" logic of CVRP, encouraging ants to build 
routes that connect nodes where the detour cost is low relative to the trip 
cost, which is a strong structural prior for vehicle routing problems.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix based on Clarke-Wright Savings.
    
    The heuristic is defined as:
    eta[i, j] = max(Savings(i, j), 0)
    
    Where Savings(i, j) = d[0, i] + d[0, j] - d[i, j]
    
    We also incorporate a capacity-aware filter: if the demand of i and j 
    combined exceeds capacity, the edge is less useful for a single route 
    (though ACO handles capacity via feasibility, we can bias against 
    'heavy' merges if we want, but standard CW savings doesn't consider 
    demand directly. However, for the heuristic matrix, we just want 
    attractive edges. Edges with high savings are attractive.
    
    To ensure numerical stability and proper scaling for ACO:
    1. Compute raw savings.
    2. Clip negative savings to 0 (or a small epsilon).
    3. Normalize or scale to be in a reasonable range for pheromone * heuristic.
    4. Handle the depot row/col specifically if needed, but usually 
       savings from depot is just the distance, which is fine.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. Compute Savings Matrix
    # d0_i = distance from depot to i
    # d0_j = distance from depot to j
    # d_ij = distance from i to j
    # Savings[i, j] = d0_i + d0_j - d_ij
    
    d0 = distance_matrix[0, :]  # Vector of distances from depot
    # Outer sum: d0_i + d0_j
    # d0[:, None] is (n, 1), d0[None, :] is (1, n)
    outer_sum = d0[:, None] + d0[None, :]
    
    # Raw savings
    savings = outer_sum - distance_matrix
    
    # 2. Clean up savings
    # Negative savings mean it's cheaper to do two trips than one merge.
    # We don't want to encourage these merges strongly.
    # Set negative savings to a small positive value to keep them "available" 
    # but unattractive, or 0? 
    # If 0, they become 1e-9 after the "not positive" check in the task contract.
    # This effectively disables edges that don't save distance.
    # This is a strong filter. Let's use a small epsilon for negative savings 
    # to avoid hard zeros if we want to allow exploration, but the task says 
    # "Heuristic entries that are not positive become 1e-9".
    # So if we return 0 or negative, they become 1e-9.
    # If we return positive, they stay.
    
    # We want to emphasize high savings. 
    # Let's shift and scale.
    # Min savings might be negative. Max savings is roughly 2 * max_dist.
    
    # To make the heuristic more robust, let's add a baseline inverse distance 
    # component? No, let's stick to pure savings first to see the effect.
    # But pure savings might be 0 for many pairs if the triangle inequality 
    # is tight or if they are far apart.
    # Actually, for Euclidean distances, d[i,j] <= d[i,0] + d[0,j] always holds.
    # So savings are always >= 0.
    # Wait, d[i,j] <= d[i,0] + d[0,j] is the triangle inequality.
    # So d[i,0] + d[0,j] - d[i,j] >= 0.
    # So savings are always non-negative.
    
    # However, if i and j are on opposite sides of the depot, d[i,j] is large, 
    # close to d[i,0] + d[0,j], so savings are small.
    # If i and j are close to each other and near the depot, savings are large.
    
    # So high savings = "good" merge.
    
    # Let's apply a power law to sharpen the difference between good and bad merges.
    # eta = savings^alpha
    # alpha > 1 makes high savings dominate more.
    # alpha < 1 flattens.
    # Let's try alpha = 1.0 first (linear), maybe 2.0 for stronger bias.
    # The current program used 1/dist. Savings are roughly proportional to dist 
    # for nearby nodes? No, savings ~ 2*dist_to_depot - dist_between.
    
    # Let's normalize savings to [0, 1] or similar to avoid scale issues with pheromones.
    # Max savings is at most 2 * max(d0).
    
    max_savings = np.max(savings)
    if max_savings > 0:
        # Scale so max is 1.0? Or keep raw?
        # Pheromone starts at 1. If heuristic is 100, prob is dominated by heuristic.
        # If heuristic is 1.0, pheromone has more influence initially.
        # Let's scale to [1, 10] range to keep pheromone relevant but guided.
        
        # Normalize
        norm_savings = savings / max_savings
        
        # Map to [1, 10]
        # eta = 1 + 9 * norm_savings
        heuristic = 1.0 + 9.0 * norm_savings
        
    else:
        heuristic = np.ones((n, n))
        
    # 3. Handle the Depot
    # For the depot (node 0), savings(0, j) = d[0,0] + d[0,j] - d[0,j] = 0.
    # So heuristic[0, j] = 1.0.
    # heuristic[i, 0] = 1.0.
    # This means leaving/returning to depot is neutral (value 1).
    # Moving between customers is > 1 if it saves distance.
    # This seems correct. The ant prefers moving between customers that save distance 
    # over going back to the depot (which has value 1).
    # Wait, if heuristic[i, 0] is 1.0 and heuristic[i, j] is 5.0, the ant prefers j.
    # This encourages staying on the route as long as savings are positive.
    # This is good! It discourages premature returns to depot.
    
    # 4. Safety Checks
    # Ensure no non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Optional: Add a small inverse distance component to break ties or 
    # handle cases where savings are similar?
    # Let's keep it pure Savings for now to test the structural change.
    
    return heuristic
# EVOLVE-BLOCK-END
