"""Different strategy: global Clarke-Wright savings pre-ordering with sequential insertion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

# Cache for savings list per (depot, demands tuple, distance_matrix id)
# Since instances are fixed per evaluation run, we cache by a hash of the problem.
_savings_cache = {}

def _get_savings_list(depot, demands, distance_matrix):
    """Compute Clarke-Wright savings list once and cache it.
    savings[i][j] = d(i, depot) + d(depot, j) - d(i, j) for i != j
    We return a list of (saving, i, j) sorted descending.
    """
    # Use a simple cache key
    n = len(demands)
    key = (depot, tuple(demands), id(distance_matrix))
    if key in _savings_cache:
        return _savings_cache[key]
    
    d = distance_matrix
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = d[i, depot] + d[depot, j] - d[i, j]
            savings.append((s, i, j))
    
    # Sort by savings descending
    savings.sort(key=lambda x: -x[0])
    _savings_cache[key] = savings
    return savings

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

def _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Clarke-Wright savings-based sequential insertion:
    1. Get precomputed savings list.
    2. Find the best (highest saving) pair (i, j) where:
       - One of i or j is the current_node (or depot if current is depot)
       - The other is in unvisited_nodes and feasible
       - The savings value is positive (worth taking)
    3. If no such pair exists, fall back to nearest feasible.
    4. If no feasible nodes at all, return depot.
    
    The key insight: we're building a route sequentially. The "savings" 
    criterion tells us which two nodes are most worth connecting. Since 
    we're at current_node, we want to connect to a node j such that 
    savings(current_node, j) is maximized, where savings is precomputed.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time check
    if _time_remaining() < 0.5:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # Get savings list
    savings_list = _get_savings_list(depot, demands, distance_matrix)
    
    d = distance_matrix
    feasible_set = set(feasible)
    
    # Strategy: find the best node to visit next based on savings.
    # For each feasible node j, compute the "effective savings" of visiting j now:
    # If we're at current_node (c), the savings of connecting c->j is:
    #   s(c, j) = d(c, depot) + d(depot, j) - d(c, j)  [if c != depot]
    #   s(depot, j) = d(depot, j) + d(depot, j) - d(depot, j) = d(depot, j) [if c == depot]
    # 
    # But more practically for sequential routing: we want to minimize the 
    # incremental cost. The savings heuristic suggests that connecting nodes 
    # with high savings reduces total tour length. 
    #
    # Alternative framing: score each feasible node j by:
    #   score(j) = d(current, j) - lambda * s(current, j)
    # where s(current, j) is the savings value. Lower score = better.
    #
    # Actually, let's use a cleaner approach:
    # The classical savings method builds routes by merging. In sequential 
    # context, we pick j that maximizes the "regret" of NOT visiting j now,
    # which is approximated by: d(current, j) is small AND d(j, depot) is large
    # (meaning j is far from depot, so we should visit it while we're out).
    #
    # Score: d(current, j) + alpha * d(j, depot)
    # Lower is better. This encourages visiting nearby nodes that are far from depot.
    
    d_from_current = d[current_node, feasible]
    d_to_depot = d[feasible, depot]
    
    # Normalize
    max_dc = np.max(d_from_current) if len(d_from_current) > 0 else 1.0
    max_dd = np.max(d_to_depot) if len(d_to_depot) > 0 else 1.0
    if max_dc == 0:
        max_dc = 1.0
    if max_dd == 0:
        max_dd = 1.0
    
    # Weighted score
    alpha = 0.3  # Mild depot penalty
    scores = d_from_current / max_dc + alpha * d_to_depot / max_dd
    
    # Additionally, use savings list to boost nodes that form high-savings pairs
    # with the current node. Find savings involving current_node.
    # Build a dict: for each node j, max savings involving current_node and j
    savings_boost = np.zeros(len(feasible))
    
    # Efficiently: for each feasible j, check if (current, j) or (j, current) 
    # has high savings. We can precompute a savings matrix for the current node.
    # But that's O(n) per call. Instead, use the precomputed list and scan.
    # For speed, just use the distance-based score.
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    best_node = int(feasible[best_idx])
    
    # Lookahead refinement: for top-3 candidates, do a 2-step greedy lookahead
    top_k = min(3, len(feasible))
    sorted_idx = np.argsort(scores)[:top_k]
    candidates = [int(feasible[i]) for i in sorted_idx]
    
    best_cost = float('inf')
    best_final = best_node
    
    for cand in candidates:
        sim_remaining = rest_capacity - demands[cand]
        sim_current = cand
        sim_cost = d[current_node, cand]
        sim_unvisited = [n for n in unvisited_nodes if n != cand]
        
        # 2-step greedy lookahead
        for _ in range(2):
            if not sim_unvisited:
                break
            sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
            if not sim_feasible:
                break
            d_sim = d[sim_current, sim_feasible]
            next_node = sim_feasible[np.argmin(d_sim)]
            sim_cost += d[sim_current, next_node]
            sim_current = next_node
            sim_remaining -= demands[next_node]
            sim_unvisited.remove(next_node)
        
        sim_cost += d[sim_current, depot]
        
        if sim_cost < best_cost:
            best_cost = sim_cost
            best_final = cand
    
    return int(best_final)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        return _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

    if len(unvisited_nodes) == 0:
        return depot

    return _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
