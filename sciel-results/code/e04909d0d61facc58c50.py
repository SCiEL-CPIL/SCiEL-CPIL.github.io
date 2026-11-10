"""Hybrid savings + lookahead heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

# Global cache for pre-computed arrays to avoid repeated conversions
_dm_cache_key = None
_dm_np = None
_demand_np = None

def _get_np_dm(distance_matrix):
    global _dm_cache_key, _dm_np
    key = id(distance_matrix)
    if _dm_cache_key != key or _dm_np is None:
        _dm_np = np.asarray(distance_matrix, dtype=float)
        _dm_cache_key = key
    return _dm_np

def _get_np_demand(demands):
    global _demand_np
    if _demand_np is None:
        _demand_np = np.asarray(demands, dtype=float)
    return _demand_np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        # Fast nearest neighbor fallback
        dm = _get_np_dm(distance_matrix)
        return int(feasible[np.argmin(dm[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited], dtype=bool)
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    dm = _get_np_dm(distance_matrix)
    all_demands = _get_np_demand(demands)

    # If at depot, use nearest neighbor to establish direction
    if current_node == depot:
        dists = dm[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # Compute key distances
    d_curr_depot = dm[current_node, depot]
    d_depot_f = dm[depot, feasible_nodes]
    d_curr_f = dm[current_node, feasible_nodes]
    
    # Savings: d_curr_depot + d_depot_j - d_curr_j
    savings = d_curr_depot + d_depot_f - d_curr_f

    # If best savings is negative, it's cheaper to go back to depot
    if np.max(savings) < 0:
        return depot

    # Normalize savings and distances for scoring
    sav_min, sav_max = np.min(savings), np.max(savings)
    sav_range = sav_max - sav_min if sav_max > sav_min else 1.0
    norm_savings = (savings - sav_min) / sav_range

    dist_min, dist_max = np.min(d_curr_f), np.max(d_curr_f)
    dist_range = dist_max - dist_min if dist_max > dist_min else 1.0
    norm_dist = (d_curr_f - dist_min) / dist_range

    # Capacity tightness: when rest_capacity is small relative to remaining demand,
    # prefer nodes that leave more room (lower demand) to avoid dead-ends
    total_remaining_demand = float(np.sum(all_demands[unvisited]))
    cap_ratio = rest_capacity / max(total_remaining_demand, 1.0)
    cap_penalty_weight = 0.35 * max(0.0, 1.0 - cap_ratio)

    # Demand normalization for capacity penalty
    dem_f = all_demands[feasible_nodes]
    dem_min, dem_max = np.min(dem_f), np.max(dem_f)
    dem_range = dem_max - dem_min if dem_max > dem_min else 1.0
    norm_dem = (dem_f - dem_min) / dem_range

    # Generate multiple candidate indices using different scoring strategies
    candidates = []

    # Candidate 1: Max raw savings
    candidates.append(np.argmax(savings))

    # Candidate 2: Savings - 0.35 * Distance
    score2 = norm_savings - 0.35 * norm_dist
    candidates.append(np.argmax(score2))

    # Candidate 3: Savings - 0.2 * Distance
    score3 = norm_savings - 0.2 * norm_dist
    candidates.append(np.argmax(score3))

    # Candidate 4: Nearest neighbor
    candidates.append(np.argmin(d_curr_f))

    # Candidate 5: Savings - 0.35*Distance - capacity penalty on demand
    score5 = norm_savings - 0.35 * norm_dist - cap_penalty_weight * norm_dem
    candidates.append(np.argmax(score5))

    # Candidate 6: Savings - 0.2*Distance - capacity penalty
    score6 = norm_savings - 0.2 * norm_dist - cap_penalty_weight * norm_dem
    candidates.append(np.argmax(score6))

    # Remove duplicates while preserving order
    unique_candidates = list(dict.fromkeys(candidates))

    if len(unique_candidates) == 0:
        return depot

    # Lookahead: Simulate route completion for each candidate using a fast
    # raw-savings greedy completion (no per-step normalization for speed),
    # then compare predicted total route length.
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])

    for idx in unique_candidates:
        if time.time() > _DEADLINE:
            break

        node = int(feasible_nodes[idx])
        
        # State after picking this node
        curr = node
        cap = rest_capacity - all_demands[node]
        remaining = set(unvisited)
        remaining.discard(node)
        
        route_cost = dm[current_node, node]
        
        # Fast greedy completion: maximize raw savings (no normalization)
        # This is faster and avoids distortion from per-step rescaling.
        while remaining and cap > 0:
            rem_arr = np.array(list(remaining), dtype=int)
            rem_demands = all_demands[rem_arr]
            feasible_mask = rem_demands <= cap
            feasible_rem = rem_arr[feasible_mask]
            
            if len(feasible_rem) == 0:
                break
            
            # Raw savings: prefer nodes that reduce total route length most
            d_curr_dep = dm[curr, depot]
            d_dep_f = dm[depot, feasible_rem]
            d_curr_f = dm[curr, feasible_rem]
            sav = d_curr_dep + d_dep_f - d_curr_f
            
            # Small proximity bonus to break ties and avoid long detours
            # Scale proximity by a small factor relative to typical savings magnitude
            sav_scale = float(np.max(np.abs(sav))) if np.max(np.abs(sav)) > 0 else 1.0
            prox_bonus = 0.15 * sav_scale * (1.0 - d_curr_f / max(np.max(d_curr_f), 1e-9))
            
            combined = sav + prox_bonus
            best_local = int(np.argmax(combined))
            best_next = int(feasible_rem[best_local])
            
            route_cost += dm[curr, best_next]
            curr = best_next
            cap -= all_demands[curr]
            remaining.discard(curr)
        
        # Cost to return to depot
        route_cost += dm[curr, depot]
        
        # We want to minimize this estimated cost
        if route_cost < best_score:
            best_score = route_cost
            best_node = node

    return best_node
# EVOLVE-BLOCK-END
