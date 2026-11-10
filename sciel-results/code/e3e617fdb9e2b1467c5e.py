"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero
    base_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Clarke-Wright Savings Heuristic
    # Calculate savings for merging two routes (i, 0, j) into (i, j)
    # Saving = d(0, i) + d(0, j) - d(i, j)
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings[i, j] = s
            savings[j, i] = s
            
    # Get list of edges sorted by savings in descending order
    # We only care about customer-customer edges for merging
    edges_to_consider = []
    for i in range(1, n):
        for j in range(i + 1, n):
            edges_to_consider.append((savings[i, j], i, j))
            
    edges_to_consider.sort(reverse=True)
    
    # Greedy construction of a reference solution using savings
    # State: Each customer starts as its own route
    # route_id: which route a customer belongs to
    # route_load: total demand in each route
    # route_customers: list of customers in each route (ordered is complex, so we just track membership)
    # Actually, for simplicity and speed in heuristic generation, we just want to identify
    # which edges are "good" merges.
    
    # We will simulate the savings algorithm to find a set of "valid" high-savings edges
    # that form a valid set of routes.
    
    route_id = np.arange(n) # Initially each node is its own route
    route_load = np.copy(demands)
    
    # For each route, we need to know the ends? 
    # Standard savings algorithm requires knowing the ends of the routes.
    # Let's simplify: We will just collect the top K savings edges that satisfy:
    # 1. i and j are in different routes
    # 2. load(route_i) + load(route_j) <= capacity
    # 3. i and j are not both already connected to other nodes in a way that breaks the path?
    #    Actually, if we just boost edges, the ACO will handle the path formation.
    #    We just want to boost edges that are likely to be in a good solution.
    
    # Let's run a simple greedy merge to find a valid set of edges.
    # We need to track the "ends" of each route to know if we can merge.
    # This is complex to implement perfectly in a heuristic pre-computation step without
    # full route representation.
    
    # Alternative: Just boost edges with high savings where the combined demand fits.
    # This ignores the "ends" constraint but is a strong signal.
    
    ref_edges = set()
    
    # To be more robust, let's implement a simplified savings that tracks route ends.
    # route_ends[route_idx] = [start_node, end_node]
    # Initially, each customer i is a route [i, i]? No, a route is a sequence.
    # If we merge i and j, the new route ends depend on orientation.
    # This is getting complicated for a heuristic booster.
    
    # Let's stick to the "High Savings + Capacity Feasible" heuristic.
    # Iterate through sorted savings.
    # If i and j are in different routes AND load[route_i] + load[route_j] <= capacity:
    #    Add edge (i, j) to ref_edges
    #    Merge routes
    
    # We need to track which route each node is in, and the load of that route.
    # And we need to track if the merge is valid topologically?
    # For the purpose of boosting, topological validity is less critical than
    # capacity and savings magnitude. The ACO will navigate the graph.
    # However, boosting an edge that *cannot* be part of a valid path (e.g. if it
    # creates a cycle in the reference logic) is bad. But since ACO builds paths
    # sequentially, it won't create cycles in the same way.
    
    # Let's just use the capacity constraint and savings value.
    
    route_of_node = np.arange(n)
    route_loads = np.copy(demands)
    
    for s_val, i, j in edges_to_consider:
        if s_val <= 0:
            break
            
        ri = route_of_node[i]
        rj = route_of_node[j]
        
        if ri == rj:
            continue
            
        # Check capacity
        if route_loads[ri] + route_loads[rj] <= capacity:
            # This is a potentially good edge
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            
            # Merge routes
            # Union-Find style or simple list update
            # For efficiency, we can just update a dictionary or array
            # Since n=50, simple array updates are fine
            # We need to update route_of_node for all nodes in rj to ri
            # And update route_loads[ri]
            
            # Find all nodes in route rj
            nodes_in_rj = np.where(route_of_node == rj)[0]
            route_of_node[nodes_in_rj] = ri
            route_loads[ri] += route_loads[rj]
            route_loads[rj] = 0 # Invalidate
            
    # Boost the heuristic values for these edges
    # Use a moderate boost to guide without dominating
    boost_factor = 10.0
    for i, j in ref_edges:
        base_heur[i, j] *= boost_factor
        base_heur[j, i] *= boost_factor

    # Ensure no non-positive or non-finite values
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
# EVOLVE-BLOCK-END
