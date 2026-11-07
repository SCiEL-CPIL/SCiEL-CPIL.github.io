# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Demand-Aware Spatial Potential strategy.
    
    The heuristic H[i, j] is defined as:
    H[i, j] = Priority(j) * Spatial_Alignment(i, j) * Capacity_Feedback(j)
    
    1. Priority(j): Higher demand customers are prioritized to ensure they are placed
       while capacity is still high (reducing the risk of them becoming infeasible later).
       Priority = 1 / (1 + demand[j]) is often used, but here we use demand[j] directly
       scaled, because in ACO, larger heuristic means more attractive. We want high
       demand to be attractive early. However, if we make high demand always attractive,
       we might pack heavy items early and then struggle with small items? 
       Actually, in VRP, it's often better to serve heavy items first or cluster them.
       Let's use a normalized demand score.
       
    2. Spatial_Alignment(i, j): Rewards moves that continue the current direction of travel.
       This reduces "zig-zag" inefficiencies. 
       Let v_i = coordinates[i] - coordinates[0] (vector from depot to i)
       Let v_j = coordinates[j] - coordinates[0]
       Alignment = dot(v_i, v_j) / (|v_i| * |v_j| + eps)
       We normalize this to be positive.
       
    3. Capacity_Feedback(j): A penalty if demand[j] is too large relative to the 
       "expected" remaining capacity. Since we don't know the path, we use the average
       remaining demand. A simpler proxy: Penalize moves to nodes with demand > capacity/2 
       unless the node is the depot or very close? 
       Actually, a better structural change: Use the "Savings" concept but modulated by 
       demand.
       
    Let's try a different structural angle: **Angle-Based Clustering Potential**.
    In VRP, routes tend to follow angular sectors around the depot.
    H[i, j] should be high if i and j are in the same angular sector relative to the depot.
    This encourages the ant to complete a "sector" before switching, which is a known
    strong heuristic for VRP (Sweep Algorithm logic).
    
    Formula:
    1. Compute angle theta[k] = atan2(y[k], x[k]) for all k.
    2. For each pair (i, j), compute the angular difference delta_theta.
    3. Attractiveness is proportional to cos(delta_theta) (high if same direction).
    4. Combine with inverse distance to prefer shorter links within the sector.
    5. Incorporate demand: Prefer moving to nodes with demand that fits "well".
       We can use a "fill" metric: How well does demand[j] fit into a "typical" remaining 
       load? This is hard to pre-compute. 
       
    Let's stick to the **Angular Sector + Distance** hybrid, which is structurally 
    different from Savings (which is triangle-inequality based) and different from 
    pure Distance.
    
    H[i, j] = (1 / d(i, j)) * (1 + Angular_Similarity(i, j)) * Demand_Weight(j)
    
    Angular_Similarity: cos(theta_i - theta_j) mapped to [0, 1].
    Demand_Weight: We want to encourage visiting high demand nodes when we are "full" 
    or "starting"? 
    Actually, a common heuristic is to prefer nodes with demand that is a "good fit" 
    for the remaining capacity. Since we can't know remaining capacity, we can use 
    the average demand of unvisited nodes? No, that's dynamic.
    
    Let's use a **Static Demand-Adjusted Savings**.
    Standard Savings: S(i, j) = d(0, i) + d(0, j) - d(i, j).
    Problem: S(i, j) ignores demand. It might merge two nodes that don't fit in one truck.
    Correction: Scale savings by the probability that they fit? 
    Or, penalize savings if demand[i] + demand[j] > capacity.
    
    New Strategy: **Capacity-Constrained Savings with Angular Bias**.
    1. Compute base savings S(i, j).
    2. If demand[i] + demand[j] > capacity, set S(i, j) = 0 (or very low).
    3. Add an angular bonus: If i and j are close in angle, boost the score.
    4. Normalize.
    
    This is structurally different from the current pure Savings because it explicitly
    zeros out infeasible merges in the heuristic, guiding the ACO away from dead-ends 
    earlier. The current program uses max(S, 0), which allows infeasible pairs to have
    high heuristic values, relying on the ACO's feasibility check to reject them. 
    If the heuristic is high for an infeasible pair, the ant might waste time considering 
    it (though it will be rejected). More importantly, if the heuristic is high for 
    infeasible pairs, it distorts the pheromone update dynamics if the ant gets stuck.
    By zeroing them out, we focus the search on feasible merges.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 290.0

    # Ensure inputs are numpy arrays
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # 1. Compute Angles relative to depot (0,0)
    # Handle depot itself (angle 0)
    angles = np.zeros(n)
    for k in range(1, n):
        x, y = coordinates[k]
        angles[k] = np.arctan2(y, x)
        
    # 2. Compute Angular Similarity
    # We want a matrix A[i, j] that is high if i and j are in the same sector.
    # cos(theta_i - theta_j) ranges [-1, 1]. Map to [0, 1] via (1 + cos)/2.
    theta_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
    angular_sim = (1.0 + np.cos(theta_diff)) / 2.0
    
    # 3. Compute Base Savings
    depot_dist = distance_matrix[0, :]
    savings = depot_dist[np.newaxis, :] + depot_dist[:, np.newaxis] - distance_matrix
    
    # 4. Capacity Feasibility Mask
    # If demand[i] + demand[j] > capacity, this merge is impossible in a single visit sequence
    # in the context of "savings" (which implies i and j are on the same route segment).
    # Note: In the actual ACO, the ant checks remaining capacity. But for the heuristic,
    # we can bias towards pairs that *could* potentially be on the same route.
    # However, even if demand[i]+demand[j] <= capacity, they might not be on the same route 
    # if other demands are large. But this is a good first-order filter.
    demand_sum = demands[:, np.newaxis] + demands[np.newaxis, :]
    feasible_pair = demand_sum <= capacity
    
    # 5. Construct Heuristic
    # H[i, j] = Savings * Angular_Sim * Feasibility_Boost
    # If infeasible, we set it to a very low value (1e-9) to discourage the ant from 
    # considering it as a "good" link, even though it will be rejected by capacity check.
    # This helps the ACO focus pheromone on feasible links.
    
    # We also want to ensure H is positive.
    # Savings can be negative (triangle inequality violation, though rare with Euclidean).
    # We use max(Savings, 0).
    
    base_heuristic = np.maximum(savings, 0.0)
    
    # Apply Angular Weighting
    # We want to strongly encourage angular continuity.
    # H = base_heuristic * (1.0 + alpha * angular_sim)
    # Let's try alpha = 1.0, so max factor is 2.0.
    weighted_heuristic = base_heuristic * (1.0 + angular_sim)
    
    # Apply Feasibility Mask
    # If the pair is infeasible (demand sum > capacity), we drastically reduce the heuristic.
    # We don't set to 0 because the ACO might still need to visit them, just not necessarily
    # as a "savings" pair. But in the context of "attractiveness of edge i->j", 
    # if i and j cannot be on the same route, the "savings" logic doesn't apply.
    # However, the ant might go i->j if they are on different routes? No, the ant constructs 
    # a single route. If it goes i->j, they are on the same route. So if demand[i]+demand[j] > capacity, 
    # it is IMPOSSIBLE for the ant to visit j immediately after i (or i after j) in a valid route 
    # where they are consecutive? 
    # Wait. The ant visits i, then j. The load increases. 
    # If demand[i] + demand[j] > capacity, it doesn't mean they can't be on the same route, 
    # it just means they can't be the *only* two, or that the specific transition i->j 
    # might be part of a route that already has load. 
    # Actually, the feasibility check in the ACO is: remaining_capacity >= demand[j].
    # It does NOT check demand[i] + demand[j] <= capacity. 
    # It checks if j fits in the *current* load.
    # So, a pair (i, j) with high demand sum might still be visited consecutively 
    # if the truck was empty before i? No, if it was empty, load after i is demand[i]. 
    # Then it needs demand[j] <= capacity - demand[i]. 
    # So demand[i] + demand[j] <= capacity IS the condition for visiting j immediately after i 
    # starting from an empty truck. 
    # But the truck might not be empty. 
    # However, if demand[i] + demand[j] > capacity, it is IMPOSSIBLE to visit j immediately 
    # after i IF the truck was empty before i. 
    # If the truck had load L, then L + demand[i] + demand[j] <= capacity. 
    # So if demand[i] + demand[j] > capacity, it is ALWAYS infeasible to visit j immediately 
    # after i, regardless of prior load (since prior load >= 0).
    # Therefore, edges between high-demand pairs that sum > capacity are ALWAYS infeasible 
    # for the transition i->j (and j->i).
    # We should set H[i, j] = 1e-9 for these pairs to save the ACO from wasting probability 
    # mass on impossible moves.
    
    heuristic_matrix = np.where(feasible_pair, weighted_heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    
    # Final sanity check
    if not np.all(np.isfinite(heuristic_matrix)):
        heuristic_matrix = np.ones_like(distance_matrix) * 1e-9
        
    return heuristic_matrix
# EVOLVE-BLOCK-END
