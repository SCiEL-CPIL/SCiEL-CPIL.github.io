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
    
    # Standard savings
    savings = d_current_depot + d_depot_feasible - d_current_feasible
    
    if np.max(savings) < 0:
        return depot

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
    cap_penalty_weight = 0.5 * max(0.0, 1.0 - cap_ratio) # Increased weight slightly

    # Generate candidates
    candidates = []
    
    # 1. Max Savings
    idx1 = np.argmax(savings)
    candidates.append(idx1)
    
    # 2. Savings - Distance
    score2 = norm_savings - norm_dist
    idx2 = np.argmax(score2)
    candidates.append(idx2)
    
    # 3. Nearest Neighbor
    idx4 = np.argmin(d_current_feasible)
    candidates.append(idx4)
    
    # 4. Savings - Distance - Capacity Penalty
    score5 = norm_savings - norm_dist - cap_penalty_weight * norm_d
    idx5 = np.argmax(score5)
    candidates.append(idx5)

    # Remove duplicates
    unique_candidates = list(dict.fromkeys(candidates))
    
    if len(unique_candidates) == 0:
        return depot

    # Limit lookahead to top 3 candidates
    unique_candidates = unique_candidates[:3]
    
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])
    
    for idx in unique_candidates:
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        node = int(feasible_nodes[idx])
        
        # State after picking this node
        curr = node
        cap = rest_capacity - all_demands[node]
        
        # Cost so far
        cost = dm[current_node, node]
        
        # 2-Step Lookahead
        # Find best next node v2 from remaining feasible nodes
        remaining = set(unvisited_nodes)
        remaining.discard(node)
        
        if remaining and cap > 0:
            rem_arr = np.array(list(remaining), dtype=int)
            rem_demands = all_demands[rem_arr]
            feasible_mask_rem = rem_demands <= cap
            feasible_rem = rem_arr[feasible_mask_rem]
            
            if len(feasible_rem) > 0:
                d_curr_dep = dm[curr, depot]
                d_dep_f = dm[depot, feasible_rem]
                d_curr_f = dm[curr, feasible_rem]
                sav = d_curr_dep + d_dep_f - d_curr_f
                
                # Pick best v2
                # Score: Savings - Distance
                max_sav = np.max(sav)
                min_d_f = np.min(d_curr_f)
                max_d_f = np.max(d_curr_f)
                if max_d_f > min_d_f:
                    norm_d_f = (d_curr_f - min_d_f) / (max_d_f - min_d_f)
                else:
                    norm_d_f = np.zeros_like(d_curr_f)
                
                # Simple score for v2
                score_v2 = sav - 0.5 * d_curr_f # Use raw distance or normalized? Raw is safer for magnitude
                
                idx_v2 = np.argmax(score_v2)
                v2 = int(feasible_rem[idx_v2])
                
                cost += dm[curr, v2]
                curr = v2
                cap -= all_demands[curr]
                
                # Estimate rest of the route
                # Simple heuristic: Sum of distances from current to depot + avg distance between remaining nodes
                # Even simpler: Just add distance to depot for now, or a rough estimate
                # Let's add a rough estimate: 
                # If we have many nodes left, assume we visit them in a cluster.
                # Approximation: d(curr, depot) + (sum of demands remaining / capacity) * d(depot, avg_node)?
                # Let's just add d(curr, depot) + 0.5 * sum(d(depot, remaining))
                if remaining:
                    rem_after_v2 = list(remaining)
                    rem_after_v2.remove(v2)
                    if rem_after_v2:
                        rem_arr_after = np.array(rem_after_v2, dtype=int)
                        # Estimate: d(curr, depot) + mean(d(depot, rem)) * len(rem)
                        # This is a very rough upper bound
                        d_rem_dep = dm[depot, rem_arr_after]
                        est_rest = dm[curr, depot] + np.mean(d_rem_dep) * len(rem_arr_after)
                        cost += est_rest
                    else:
                        cost += dm[curr, depot]
                else:
                    cost += dm[curr, depot]
            else:
                cost += dm[curr, depot]
        else:
            cost += dm[curr, depot]

        if cost < best_score:
            best_score = cost
            best_node = node

    return best_node
