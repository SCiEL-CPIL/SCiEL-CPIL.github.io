# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach: Depot-pressure field with demand-aware scaling.
      1. Base heuristic: inverse distance.
      2. Depot edges (0->j): scale by 1/sqrt(demand_j) to prioritize high-demand
         nodes early when the truck is full.
      3. Customer-customer edges (i->j): scale by a capacity-potential factor
         that rewards visiting high-demand nodes when remaining capacity is high.
      4. Clarke-Wright savings boost for customer-customer edges to encourage
         route merging.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]
    
    # Safe distance matrix
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Base heuristic: inverse distance
    heuristic = 1.0 / dist_safe
    
    n_customers = n - 1
    if n_customers > 0:
        # Depot demand is 0, customer demands are 1-9
        depot_demand = demands[0]
        cust_demands = demands[1:]
        
        # --- Depot-Pressure Field ---
        # For depot->customer edges (row 0): 
        # High-demand customers should be visited early (when truck is full).
        # Scale by 1/sqrt(demand_j) so high demand -> lower multiplier? 
        # Wait: "attractiveness inversely proportional to demand" means high demand = less attractive?
        # But the directive says "high-demand nodes are harder to serve and should be visited early"
        # So we want to BOOST high-demand nodes for depot edges.
        # Let's use: boost = 1 + demand_j / max_demand, so higher demand = higher boost.
        
        max_demand = np.max(cust_demands) if np.max(cust_demands) > 0 else 1.0
        
        # Depot to customer: boost high-demand nodes
        # heuristic[0, j] *= (1 + cust_demands[j-1] / max_demand)
        # Actually, let's think about "remaining capacity potential".
        # When starting at depot, remaining capacity = capacity.
        # The "pressure" is: if demand_j is high, it's good to visit it now because
        # later you might not have enough capacity.
        # So boost = capacity / (capacity - demand_j + 1) or similar.
        # Simpler: boost proportional to demand_j.
        
        depot_boost = 1.0 + (cust_demands / max_demand) * 2.0  # Range: 1.0 to 3.0
        heuristic[0, 1:] *= depot_boost
        
        # Customer to depot edges (column 0): 
        # When returning to depot, the "pressure" is lower.
        # Keep base or slight boost for high-demand nodes (they freed up capacity).
        # heuristic[i, 0] *= 1.0 + (cust_demands[i-1] / max_demand) * 0.5
        
        # --- Customer-Customer Edges with Capacity Potential ---
        # For edge i->j, the "capacity potential" depends on:
        # - How much capacity remains after visiting i (depends on route so far, but we don't know that)
        # - Heuristic approximation: assume average remaining capacity
        # - Reward edges where j has high demand (visit heavy nodes when capacity is still high)
        
        # Savings-based boost: d_0i + d_0j - d_ij
        d_0 = dist_safe[0, :]  # Distance from depot to each node
        d_0i = d_0[1:]  # Depot to customer i
        d_0j = d_0[1:]  # Depot to customer j
        
        # Savings matrix for customers
        # savings[i,j] = d_0i + d_0j - d_ij
        d_cust = dist_safe[1:, 1:]
        savings = d_0i[:, None] + d_0j[None, :] - d_cust
        
        # Normalize savings
        max_savings = np.max(savings) if np.max(savings) > 0 else 1.0
        savings_norm = np.maximum(savings, 0) / max_savings
        
        # Demand efficiency: visiting high-demand j is better
        # capacity_potential[j] = 1 + demand_j / max_demand
        cap_potential = 1.0 + (cust_demands / max_demand) * 1.5  # Range: 1.0 to 2.5
        
        # Combine: boost[i,j] = (1 + savings_norm[i,j]) * cap_potential[j]
        # Apply to customer-customer edges
        cc_boost = (1.0 + savings_norm) * cap_potential[None, :]
        heuristic[1:, 1:] *= cc_boost
        
        # Customer to depot: slight boost for high-demand nodes
        # heuristic[i, 0] *= 1.0 + (cust_demands[i-1] / max_demand) * 0.5
        cust_to_depot_boost = 1.0 + (cust_demands / max_demand) * 0.5
        heuristic[1:, 0] *= cust_to_depot_boost

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
