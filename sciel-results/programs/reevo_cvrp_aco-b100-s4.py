"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Base heuristic: Inverse distance with a power to emphasize shorter edges
    # A power > 1 makes the heuristic more "local", which often helps ACO
    # avoid long, unnecessary jumps.
    power = 1.5
    eps = 1e-9
    base_heur = 1.0 / (distance_matrix + eps) ** power
    
    # Generate a reference set of edges using a Savings Heuristic (Clarke-Wright)
    # This approach is different from Nearest Neighbor as it optimizes for merging routes.
    
    # Initialize each customer as its own route
    # routes: dict mapping route_id -> {'start': int, 'end': int, 'demand': float, 'customers': list}
    # cust_to_route: dict mapping customer_id -> route_id
    
    routes = {}
    cust_to_route = {}
    
    for i in range(1, n):
        routes[i] = {
            'start': i,
            'end': i,
            'demand': demands[i],
            'customers': [i]
        }
        cust_to_route[i] = i
        
    # Calculate savings for all pairs
    # savings[i][j] = d(0,i) + d(0,j) - d(i,j)
    d0 = distance_matrix[0]
    savings = d0[:, None] + d0[None, :] - distance_matrix
    
    # Create list of pairs (savings, i, j) for i < j
    pairs = []
    for i in range(1, n):
        for j in range(i + 1, n):
            pairs.append((savings[i, j], i, j))
            
    # Sort by savings descending
    pairs.sort(reverse=True)
    
    ref_edges = set()
    
    # Process savings
    for sav, i, j in pairs:
        rid_i = cust_to_route.get(i)
        rid_j = cust_to_route.get(j)
        
        if rid_i is None or rid_j is None or rid_i == rid_j:
            continue
            
        route_i = routes.get(rid_i)
        route_j = routes.get(rid_j)
        
        if route_i is None or route_j is None:
            continue
            
        # Check capacity
        if route_i['demand'] + route_j['demand'] > capacity:
            continue
            
        # Check if we can merge.
        # We can merge if i is an endpoint of route_i and j is an endpoint of route_j.
        # Endpoints are 'start' and 'end'.
        
        i_is_start = (route_i['start'] == i)
        i_is_end = (route_i['end'] == i)
        j_is_start = (route_j['start'] == j)
        j_is_end = (route_j['end'] == j)
        
        # We need one to be 'start' and the other 'end' to connect them directly
        # without reversing the internal order of the route (which would change other edges).
        # However, reversing a route just swaps start and end and reverses the customer list.
        # For the purpose of edge extraction, we just need to know which edge connects them.
        
        # Case 1: i is end of A, j is start of B
        # Connect i -> j
        if i_is_end and j_is_start:
            # Merge A and B
            new_start = route_i['start']
            new_end = route_j['end']
            new_demand = route_i['demand'] + route_j['demand']
            new_customers = route_i['customers'] + route_j['customers']
            
            routes[rid_i] = {
                'start': new_start,
                'end': new_end,
                'demand': new_demand,
                'customers': new_customers
            }
            for c in new_customers:
                cust_to_route[c] = rid_i
            del routes[rid_j]
            
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            continue # Move to next pair
            
        # Case 2: i is start of A, j is end of B
        # Connect i -> j (by reversing one of the routes)
        if i_is_start and j_is_end:
            # We can reverse route J so that j becomes start.
            # Or reverse route I so that i becomes end.
            # Let's just record the edge (i, j). The internal structure doesn't matter for the heuristic matrix.
            # But we must update the route info correctly for future merges.
            # If we reverse route J, its start becomes j and end becomes old_start.
            # Then we have i (start of A) and j (start of new J). Still Start-Start.
            # We need End-Start.
            # So we must reverse one route to make i an End or j a Start.
            # If we reverse A: i becomes End. j is End of B. End-End.
            # If we reverse B: j becomes Start. i is Start of A. Start-Start.
            # If we reverse BOTH: i becomes End, j becomes Start. End-Start.
            
            # So, if i is Start of A and j is End of B, we can merge by reversing both.
            # New A: start=old_end_A, end=i, customers=reversed(old_customers_A)
            # New B: start=j, end=old_start_B, customers=reversed(old_customers_B)
            # Now i is End of New A, j is Start of New B.
            
            new_start_i = route_i['end']
            new_end_i = i
            new_customers_i = list(reversed(route_i['customers']))
            
            new_start_j = j
            new_end_j = route_j['start']
            new_customers_j = list(reversed(route_j['customers']))
            
            # Now merge New A (ending at i) and New B (starting at j)
            # Combined Start: new_start_i
            # Combined End: new_end_j
            # Combined Customers: new_customers_i + new_customers_j
            
            new_start = new_start_i
            new_end = new_end_j
            new_demand = route_i['demand'] + route_j['demand']
            new_customers = new_customers_i + new_customers_j
            
            routes[rid_i] = {
                'start': new_start,
                'end': new_end,
                'demand': new_demand,
                'customers': new_customers
            }
            for c in new_customers:
                cust_to_route[c] = rid_i
            del routes[rid_j]
            
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            continue
            
        # Case 3: i is end of A, j is end of B
        # Reverse B so j becomes start.
        if i_is_end and j_is_end:
            # Reverse B
            new_start_j = j
            new_end_j = route_j['start']
            new_customers_j = list(reversed(route_j['customers']))
            
            # Now i is End of A, j is Start of New B
            new_start = route_i['start']
            new_end = new_end_j
            new_demand = route_i['demand'] + route_j['demand']
            new_customers = route_i['customers'] + new_customers_j
            
            routes[rid_i] = {
                'start': new_start,
                'end': new_end,
                'demand': new_demand,
                'customers': new_customers
            }
            for c in new_customers:
                cust_to_route[c] = rid_i
            del routes[rid_j]
            
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            continue
            
        # Case 4: i is start of A, j is start of B
        # Reverse A so i becomes end.
        if i_is_start and j_is_start:
            # Reverse A
            new_start_i = route_i['end']
            new_end_i = i
            new_customers_i = list(reversed(route_i['customers']))
            
            # Now i is End of New A, j is Start of B
            new_start = new_start_i
            new_end = route_j['end']
            new_demand = route_i['demand'] + route_j['demand']
            new_customers = new_customers_i + route_j['customers']
            
            routes[rid_i] = {
                'start': new_start,
                'end': new_end,
                'demand': new_demand,
                'customers': new_customers
            }
            for c in new_customers:
                cust_to_route[c] = rid_i
            del routes[rid_j]
            
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            continue

    # Boost the heuristic values for edges in the reference solution
    boost_factor = 5.0
    for i, j in ref_edges:
        base_heur[i, j] *= boost_factor
        base_heur[j, i] *= boost_factor

    # Ensure no non-positive or non-finite values
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
# EVOLVE-BLOCK-END
