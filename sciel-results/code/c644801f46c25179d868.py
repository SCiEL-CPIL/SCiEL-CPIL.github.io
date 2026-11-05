"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Fast fallback: nearest feasible neighbor."""
    if len(unvisited_nodes) == 0:
        return depot
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    d = distance_matrix[current_node, feasible]
    return int(feasible[np.argmin(d)])

def _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Refined savings heuristic:
    - Score = dist(current, node) + alpha * dist(node, depot) - beta * capacity_urgency
    - Lightweight 1-step lookahead only when capacity is tight
    - Smooth tiebreaker via log-sum-exp for stability
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time check: if tight, use fast path
    if _time_remaining() < 0.5:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # --- Core savings score ---
    d_from_current = distance_matrix[current_node, feasible]
    d_to_depot = distance_matrix[feasible, depot]

    # Classical savings: dist(current, node) + dist(node, depot)
    # This is the exact incremental cost if we go current -> node -> depot
    scores = d_from_current + d_to_depot

    # --- Capacity urgency adjustment ---
    # When capacity is tight, prefer nodes that leave more room for others
    # Penalty for high-demand nodes when capacity is low
    cap_ratio = rest_capacity / 50.0 if rest_capacity > 0 else 0.0
    if cap_ratio < 0.5:
        # Scale penalty by how tight capacity is
        urgency_scale = (0.5 - cap_ratio) / 0.5  # 0 at cap_ratio=0.5, 1 at cap_ratio=0
        demands_feasible = np.array([demands[n] for n in feasible])
        max_demand = np.max(demands_feasible) if len(demands_feasible) > 0 else 1.0
        if max_demand == 0:
            max_demand = 1.0
        norm_demand = demands_feasible / max_demand
        scores = scores + urgency_scale * 2.0 * norm_demand

    # --- Lightweight 1-step lookahead for tight capacity ---
    if cap_ratio < 0.3 and _time_remaining() > 2.0:
        # For each top candidate, estimate the cost of the next nearest feasible node
        k = min(3, len(feasible))
        sorted_idx = np.argsort(scores)
        top_candidates = [feasible[i] for i in sorted_idx[:k]]
        
        best_node = top_candidates[0]
        best_cost = float('inf')
        
        for cand in top_candidates:
            sim_remaining = rest_capacity - demands[cand]
            sim_unvisited = [n for n in unvisited_nodes if n != cand]
            sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
            
            if sim_feasible:
                # Cost to go from cand to nearest feasible, then back to depot
                d_next = distance_matrix[cand, sim_feasible]
                nearest_next = sim_feasible[np.argmin(d_next)]
                # Total: current->cand + cand->nearest + nearest->depot
                total = distance_matrix[current_node, cand] + d_next.min() + distance_matrix[nearest_next, depot]
            else:
                # No more feasible: just return to depot
                total = distance_matrix[current_node, cand] + distance_matrix[cand, depot]
            
            if total < best_cost:
                best_cost = total
                best_node = cand
        
        return int(best_node)

    # --- Smooth tiebreaker via log-sum-exp ---
    # Use soft-min to handle near-ties gracefully
    min_score = np.min(scores)
    temperature = 0.01 * (np.max(scores) - min_score + 1e-10)
    shifted = scores - min_score
    weights = np.exp(-shifted / temperature)
    weights = weights / np.sum(weights)
    
    # Select argmin of scores, but use weights as a stable tiebreaker
    sorted_indices = np.argsort(scores)
    # If top two scores are very close, use the weight to break the tie
    if len(sorted_indices) >= 2:
        s0, s1 = scores[sorted_indices[0]], scores[sorted_indices[1]]
        if abs(s0 - s1) < 1e-6:
            w0, w1 = weights[sorted_indices[0]], weights[sorted_indices[1]]
            if w1 > w0:
                best_idx = sorted_indices[1]
            else:
                best_idx = sorted_indices[0]
        else:
            best_idx = sorted_indices[0]
    else:
        best_idx = sorted_indices[0]
    
    return int(feasible[best_idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        return _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

    if len(unvisited_nodes) == 0:
        return depot

    return _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
