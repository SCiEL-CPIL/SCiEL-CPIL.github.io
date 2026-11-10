"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger = more attractive) of the same shape.
    
    Strategy:
    1. Power-law base heuristic: 1 / d^1.5 to strongly encourage local moves.
    2. Node-potential field: Combines local spatial demand density and distance to
       the demand centroid. This creates a potential landscape that guides ants
       toward dense, central regions of the customer set.
    3. Demand-clustering pressure: Boosts edges between high-demand customers to
       encourage serving them together.
    4. Depot-anchored adjustments: Specific boosts for edges leaving and entering
       the depot based on demand and proximity.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Ensure finite and positive distance matrix
    dm = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    dm = np.maximum(dm, 1e-9)
    
    # --- Step 1: Power-law Base Heuristic ---
    # Using 1.5 power to emphasize local moves more strongly than 1/d
    # This aligns with the knowledge finding that power-law base improved scores
    base = 1.0 / (dm ** 1.5)
    
    # --- Step 2: Node-Potential Field ---
    # We want to guide ants toward regions with high demand density and close to the center of demand.
    
    customers = np.arange(1, n)
    cust_coords = coords[1:]
    cust_demands = dem[1:]
    
    # 2a. Local Spatial Density
    # Compute pairwise distances between customers
    diff = cust_coords[:, None, :] - cust_coords[None, :, :]
    dist_cust = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Bandwidth for Gaussian kernel
    # Use a fraction of the typical inter-customer distance
    if len(customers) > 1:
        # Avoid zero distances in mean calculation
        non_zero_dists = dist_cust[dist_cust > 0]
        if len(non_zero_dists) > 0:
            typical_dist = np.mean(non_zero_dists)
        else:
            typical_dist = 0.5
    else:
        typical_dist = 0.5
        
    bandwidth = max(typical_dist * 0.8, 0.05)
    
    # Density at each customer: weighted sum of demands in neighborhood
    kernel = np.exp(-dist_cust ** 2 / (2 * bandwidth ** 2))
    density = kernel @ cust_demands  # shape (n-1,)
    
    # Normalize density to [0.5, 1.5] to modulate the base heuristic
    if density.max() > density.min():
        density_norm = 0.5 + 1.0 * (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.ones(n - 1)
    
    # 2b. Centroid Pressure
    # Compute demand-weighted centroid of customers
    total_demand = np.sum(cust_demands)
    if total_demand > 0:
        centroid = np.sum(cust_coords * cust_demands[:, None], axis=0) / total_demand
    else:
        centroid = coords[0]
        
    # Distance from each customer to the centroid
    dist_to_centroid = np.sqrt(np.sum((cust_coords - centroid) ** 2, axis=1))
    
    # Exponential decay pressure: higher pressure closer to centroid
    # Scale is related to typical distance
    pressure_scale = max(typical_dist * 0.5, 0.01)
    centroid_pressure = np.exp(-dist_to_centroid / pressure_scale)
    
    # Normalize centroid pressure to [0.5, 1.5]
    # The max is 1.0 (at centroid), min is exp(-max_dist/scale)
    # We want to map this to a range that doesn't overwhelm the density term
    # Let's normalize it to [0.5, 1.5] as well
    if centroid_pressure.max() > centroid_pressure.min():
        centroid_pressure_norm = 0.5 + 1.0 * (centroid_pressure - centroid_pressure.min()) / (centroid_pressure.max() - centroid_pressure.min())
    else:
        centroid_pressure_norm = np.ones(n - 1)
        
    # Combine density and centroid pressure into a single node potential
    # Potential is high where demand is dense AND close to center
    node_potential = density_norm * centroid_pressure_norm
    
    # Extend potential to full node vector (depot gets neutral value 1.0)
    potential_full = np.ones(n)
    potential_full[1:] = node_potential
    
    # --- Step 3: Apply Node Potential to Base Heuristic ---
    # heuristic[i, j] = base[i, j] * potential[j]
    heuristic = base * potential_full[None, :]
    
    # --- Step 4: Demand-Clustering Pressure ---
    # Boost edges between high-demand customers
    # This encourages the ant to visit high-demand nodes in sequence
    demand_outer = np.outer(dem, dem)
    demand_pressure = demand_outer / cap
    
    # Only apply to customer-to-customer edges
    customer_mask = np.zeros((n, n), dtype=bool)
    customer_mask[1:, 1:] = True
    
    # Add a moderate boost proportional to demand product and base attraction
    # Scale factor 0.5 to ensure it's significant but not dominant
    demand_boost_matrix = np.zeros((n, n))
    demand_boost_matrix[customer_mask] = 0.5 * demand_pressure[customer_mask] * base[customer_mask]
    
    heuristic += demand_boost_matrix
    
    # --- Step 5: Depot-Anchored Structure ---
    # 5a. Depot to Customer
    # Boost by demand (high demand = visit early) and proximity
    if n > 1:
        # Proximity is already in base[0, 1:]
        # Add demand-based priority
        demand_priority = dem[1:] / max(np.max(dem[1:]), 1.0)
        heuristic[0, 1:] += 0.3 * demand_priority * base[0, 1:]
        
    # 5b. Customer to Depot
    # Boost by proximity (closer to depot = easier return)
    # The base heuristic already handles proximity.
    # We can add a small boost to encourage returning to depot when close.
    if n > 1:
        # Base is 1/d^1.5, which is strong for close distances.
        # No additional boost needed, or a very small one.
        # Let's leave it as is to avoid over-optimizing for depot returns.
        pass
    
    # --- Final Checks ---
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Time guard
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
