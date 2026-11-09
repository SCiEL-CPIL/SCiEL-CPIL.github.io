"""Capacitated vehicle routing: pick the next customer or return to depot."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: the evaluator kills the process at 150s wall-clock.
# We stop doing heavy work once we get close to that limit.
_T0 = time.perf_counter()
_DEADLINE = _T0 + 145.0  # 5s safety margin


def _deadline_exceeded():
    return time.perf_counter() > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer to visit, or return 0 (depot) to close the route.
    
    Strategy: Clarke-Wright savings based heuristic with capacity awareness.
    The savings of visiting node j next (instead of going back to depot) is:
        saving = dist(current, depot) - dist(current, j) + dist(j, depot)
    Wait, that's not quite right. Let me think more carefully.
    
    Actually, the classic savings approach: if we go current -> depot, cost is dist(current, depot).
    If we go current -> j -> depot, cost is dist(current, j) + dist(j, depot).
    The "savings" of inserting j before returning to depot is:
        dist(current, depot) - (dist(current, j) + dist(j, depot))
    But that's negative if the triangle inequality holds strictly... 
    
    Let me reconsider. The standard Clarke-Wright savings for two routes:
    Route 1: depot -> i -> depot
    Route 2: depot -> j -> depot
    Combined: depot -> i -> j -> depot
    Savings = dist(depot,i) + dist(depot,j) - dist(i,j)
    
    For extending a current route ending at current_node, the decision is:
    - Go to depot: cost = dist(current, depot)
    - Go to node j then eventually depot: we save the trip from current to depot,
      but we pay dist(current, j) + dist(j, depot) later.
    
    A better heuristic: prioritize nodes that are "close" to the current node
    AND also close to the depot (to minimize the final return leg), while
    respecting capacity constraints.
    
    Score for node j:
        score = dist(current, j) + lambda * dist(j, depot)
    where lambda balances immediate travel vs. future return cost.
    
    Additionally, we should consider: if remaining capacity is tight, prefer
    nodes with lower demand to leave room for more nodes.
    """
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter to feasible nodes (demand <= rest_capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Deadline check: if we're close to the limit, just pick nearest feasible node
    if _deadline_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dists)
        return int(feasible_nodes[best_idx])
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    demands_feasible = demands[feasible_nodes]
    
    # Total demand of all unvisited feasible nodes
    total_demand = np.sum(demands_feasible)
    
    # Adaptive lambda: when capacity is tight relative to remaining demand,
    # weight the depot distance more heavily to close routes sooner
    capacity_ratio = rest_capacity / max(total_demand, 1.0)
    
    # lambda increases as capacity becomes tighter
    # When capacity_ratio is high (plenty of capacity), lambda is small (favor nearby nodes)
    # When capacity_ratio is low (tight capacity), lambda is larger (favor nodes close to depot)
    lambda_val = 0.3 + 0.7 * (1.0 - min(capacity_ratio, 1.0))
    
    # Primary score: balance between distance from current and distance to depot
    scores = dists_from_current + lambda_val * dists_to_depot
    
    # Additional consideration: if only a few nodes fit in remaining capacity,
    # prefer lower demand nodes to maximize the number of stops
    if rest_capacity < 2 * np.mean(demands_feasible):
        # Normalize demand to [0,1] range for penalty
        max_d = np.max(demands_feasible)
        min_d = np.min(demands_feasible)
        if max_d > min_d:
            demand_penalty = (demands_feasible - min_d) / (max_d - min_d)
        else:
            demand_penalty = np.zeros_like(demands_feasible, dtype=float)
        scores += 0.1 * demand_penalty * np.max(dists_from_current)
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
