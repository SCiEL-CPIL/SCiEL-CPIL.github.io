import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Check deadline
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Ensure numpy arrays for efficient indexing
    unvisited_arr = np.asarray(unvisited_nodes)
    
    # Get demands for these nodes
    try:
        node_demands = demands[unvisited_arr]
    except Exception:
        node_demands = np.array([demands[i] for i in unvisited_arr])

    # Feasibility: demand <= rest_capacity
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Ensure distance_matrix is numpy array
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.asarray(distance_matrix)

    d_current_depot = distance_matrix[current_node, depot]
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Standard savings: S(v) = d(current, depot) + d(depot, v) - d(current, v)
    savings = d_current_depot + d_depot_feasible - d_current_feasible
    
    # If the best saving is negative, it's cheaper to go back to depot
    # (This implies that even the "best" detour increases total cost vs closing now)
    # Note: In CVRP, we often must close if no feasible nodes remain, but here we have feasible nodes.
    # However, if going to any node is significantly worse than returning to depot, we should return.
    # The savings heuristic typically compares the cost of inserting the node into the current route
    # vs keeping them separate. 
    # If max_savings < 0, it means d(current, v) > d(current, depot) + d(depot, v).
    # This triangle inequality violation (or rather, the geometry) suggests v is "behind" us relative to depot.
    # But wait, if we return to depot, we start a new route.
    # The "savings" formulation usually assumes we are building a route from depot.
    # If we are at current_node, and we go to depot, cost is d(current, depot).
    # If we go to v then depot, cost is d(current, v) + d(v, depot).
    # Savings = (d(current, depot) + d(depot, v)) - (d(current, v) + d(v, depot)) ?? No.
    # Standard Clarke-Wright savings for merging two routes (depot-a-...-b-depot) is d(depot,a)+d(depot,b)-d(a,b).
    # Here, we are extending a route.
    # Cost to close now: d(current, depot).
    # Cost to visit v then close: d(current, v) + d(v, depot).
    # Difference: (d(current, v) + d(v, depot)) - d(current, depot).
    # We want to minimize this difference.
    # If d(current, v) + d(v, depot) < d(current, depot), it's better to visit v?
    # Unlikely in Euclidean space unless v is very close to current and depot is far.
    # Actually, the standard heuristic "Savings" is used to decide *which* node to visit *if* we are going to visit one.
    # The decision to close the route is usually based on capacity or if no feasible nodes remain.
    # However, sometimes it's better to close early to avoid a long detour.
    # Let's stick to the proven logic: if no feasible nodes, return depot.
    # The inspirations check `if np.max(savings) < 0: return depot`. Let's adopt this as a heuristic improvement.
    # It implies that if the "savings" (which measures how much we save by NOT going back to depot yet) is negative,
    # we are losing distance by staying out.
    
    if np.max(savings) < 0:
        return depot

    # Calculate geometric smoothness score
    # We want to minimize the turning angle.
    # Vector from previous node to current node? 
    # We don't have the previous node in the signature. 
    # We only have current_node. 
    # We can approximate the "direction" of the current route by the vector from depot to current_node? 
    # Or we can use the vector from the *last* node we visited? 
    # The function signature does NOT provide the previous node. 
    # It provides `current_node`. 
    # If we are at `current_node`, the "incoming" direction is unknown unless we assume the route started at depot.
    # If the route is depot -> ... -> current_node, the last segment was (prev_node -> current_node).
    # We don't have prev_node.
    # However, we can use the vector from depot to current_node as a proxy for the general direction if the route is short.
    # Or, more robustly, we can look at the angle between (current_node -> depot) and (current_node -> candidate).
    # If we go to candidate, the next segment is (current_node -> candidate).
    # Then we will likely go to depot or another node.
    # A "smooth" path avoids sharp turns.
    # Let's define the angle between the vector (current_node -> depot) and (current_node -> candidate).
    # If the angle is small, we are heading towards the depot? No, that would be a sharp turn if we were heading away.
    # Actually, if we are heading away from depot, and we pick a node that continues that direction, the angle between (depot->current) and (current->next) should be small (collinear).
    # Let v1 = current_node - depot (vector from depot to current)
    # Let v2 = candidate - current_node (vector from current to candidate)
    # We want v1 and v2 to be aligned (dot product positive, angle small).
    # Cosine similarity = (v1 . v2) / (|v1| |v2|)
    # High cosine similarity = smooth continuation.
    
    # We don't have coordinates! We only have a distance matrix.
    # We cannot compute vectors directly.
    # We can use the law of cosines to find the angle at `current_node` in the triangle (depot, current_node, candidate).
    # a = d(depot, current_node)
    # b = d(current_node, candidate)
    # c = d(depot, candidate)
    # Angle theta at current_node:
    # c^2 = a^2 + b^2 - 2ab cos(theta)
    # cos(theta) = (a^2 + b^2 - c^2) / (2ab)
    
    # We want to MINIMIZE the turning angle. 
    # If we are traveling from depot to current_node, and we go to candidate, the "turn" is the angle between the segment (depot->current) and (current->candidate).
    # Wait, the segment we just traversed was (prev -> current). We don't know prev.
    # If we assume the "ideal" direction is towards the depot (to close the route), then we want to minimize the angle between (current->depot) and (current->candidate)?
    # No, that would mean we pick nodes that are "on the way back".
    # If we want to cluster, we want to pick nodes that are in the same direction we are already going.
    # Without prev_node, we can't know the direction we are going.
    # 
    # Alternative interpretation of "smoothness" with only distance matrix:
    # The savings heuristic already captures some of this. 
    # High savings means d(current, depot) + d(depot, v) is much larger than d(current, v).
    # This implies v is "close" to current relative to the depot, i.e., we are not backtracking towards the depot.
    # 
    # Let's look at the inspiration: "minimizes the local curvature... relative to the previous segment".
    # Since we don't have the previous segment, we cannot strictly implement this.
    # However, we can use a surrogate:
    # The "savings" value is high when the triangle (depot, current, v) is "flat" in a way that v is close to current.
    # 
    # Let's refine the scoring by combining Savings and a "proximity to depot" penalty or bonus?
    # The inspiration 1 uses: `score = norm_savings - norm_dist - cap_penalty * norm_d`.
    # This is a strong candidate.
    # Let's implement a multi-candidate approach with lookahead, as in Inspiration 1, but keep it lightweight to fit in the time budget.
    
    # Pre-compute demand array for speed
    all_demands = np.array(demands, dtype=float)
    dm = distance_matrix
    
    # Normalize savings
    max_saving = np.max(savings)
    min_saving = np.min(savings)
    if max_saving > min_saving:
        norm_savings = (savings - min_saving) / (max_saving - min_saving)
    else:
        norm_savings = np.ones_like(savings)

    # Normalize distance
    max_dist = np.max(d_current_feasible)
    min_dist = np.min(d_current_feasible)
    if max_dist > min_dist:
        norm_dist = (d_current_feasible - min_dist) / (max_dist - min_dist)
    else:
        norm_dist = np.zeros_like(d_current_feasible)

    # Normalize demands
    d_j = node_demands[feasible_mask].astype(float)
    max_d = np.max(d_j)
    min_d = np.min(d_j)
    if max_d > min_d:
        norm_d = (d_j - min_d) / (max_d - min_d)
    else:
        norm_d = np.zeros_like(d_j)

    # Capacity tightness penalty
    total_remaining_demand = float(np.sum(node_demands))
    cap_ratio = rest_capacity / max(total_remaining_demand, 1.0)
    cap_penalty_weight = 0.3 * max(0.0, 1.0 - cap_ratio)

    # Generate candidates
    candidates = []
    
    # 1. Max Savings
    idx1 = np.argmax(savings)
    candidates.append(idx1)
    
    # 2. Savings - Distance
    score2 = norm_savings - norm_dist
    idx2 = np.argmax(score2)
    candidates.append(idx2)
    
    # 3. Savings - 0.5 * Distance
    score3 = norm_savings - 0.5 * norm_dist
    idx3 = np.argmax(score3)
    candidates.append(idx3)
    
    # 4. Nearest Neighbor
    idx4 = np.argmin(d_current_feasible)
    candidates.append(idx4)
    
    # 5. Savings - Distance - Capacity Penalty
    score5 = norm_savings - norm_dist - cap_penalty_weight * norm_d
    idx5 = np.argmax(score5)
    candidates.append(idx5)
    
    # Remove duplicates
    unique_candidates = list(dict.fromkeys(candidates))
    
    if len(unique_candidates) == 0:
        return depot

    # Lookahead simulation for the top candidates
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])
    
    # Limit lookahead to top 3 candidates to save time
    unique_candidates = unique_candidates[:3]
    
    for idx in unique_candidates:
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        node = int(feasible_nodes[idx])
        
        # State after picking this node
        curr = node
        cap = rest_capacity - all_demands[node]
        remaining = set(unvisited_nodes)
        remaining.discard(node)
        
        route_cost = dm[current_node, node]
        
        # Fast greedy completion: pick max savings, tie-break by nearest
        while remaining and cap > 0:
            rem_arr = np.array(list(remaining), dtype=int)
            rem_demands = all_demands[rem_arr]
            feasible_mask_rem = rem_demands <= cap
            feasible_rem = rem_arr[feasible_mask_rem]
            
            if len(feasible_rem) == 0:
                break
            
            d_curr_dep = dm[curr, depot]
            d_dep_f = dm[depot, feasible_rem]
            d_curr_f = dm[curr, feasible_rem]
            sav = d_curr_dep + d_dep_f - d_curr_f
            
            # Simple heuristic: Max savings, tie-break by min distance
            max_sav = np.max(sav)
            best_indices = np.where(sav == max_sav)[0]
            
            if len(best_indices) == 1:
                best_local = best_indices[0]
            else:
                best_nodes_local = feasible_rem[best_indices]
                d_curr_best_local = dm[curr, best_nodes_local]
                min_dist_idx = np.argmin(d_curr_best_local)
                best_local = best_indices[min_dist_idx]
                
            best_next = int(feasible_rem[best_local])
            
            route_cost += dm[curr, best_next]
            curr = best_next
            cap -= all_demands[curr]
            remaining.discard(curr)
        
        # Cost to return to depot
        route_cost += dm[curr, depot]
        
        if route_cost < best_score:
            best_score = route_cost
            best_node = node
    
    return best_node
