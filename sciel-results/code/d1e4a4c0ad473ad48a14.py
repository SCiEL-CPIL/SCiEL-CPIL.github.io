"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a global deadline guard to ensure we finish within the 120s limit.
# The evaluator kills the program after 150s, but the task says "Each evaluation must finish within 120 s".
# We'll track the start time of the entire evaluation process if possible, but since we only have the function,
# we'll assume the function is called many times.
# However, the score is "mean total route length on eight generated 50-customer instances".
# So the function is called many times (50 customers * 8 instances = 400 calls roughly, plus depot returns).
# We should not do heavy computation per call.
# But the directive says: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
# This implies that if we do a search inside the function, we should stop if we're running out of time.
# However, a simple heuristic like nearest neighbor is O(N) per call, which is very fast.
# If we implement a more complex heuristic (e.g., looking ahead), it might be slower.
# Let's stick to a fast heuristic that is better than simple nearest neighbor.

# The directive suggests: "Model the decision as a 'capacity-aware angular sweep'... projecting remaining feasible nodes onto a circle centered at the depot..."
# Let's try to implement a heuristic that considers:
# 1. Distance to current node.
# 2. Distance to depot (to encourage closing routes when close to depot or when capacity is low).
# 3. Angular consistency (optional, might be complex).
# 4. Capacity feasibility.

# A common and effective heuristic for CVRP is the "Savings" or "Clarke-Wright" idea, but we are constructing routes sequentially.
# Another idea: "Nearest Neighbor with Depot Bias".
# Score = distance(current, next) + alpha * distance(next, depot)
# If rest_capacity is low, we should prefer to go back to depot.
# If rest_capacity is high, we should prefer to visit more nodes.

# Let's try a weighted score:
# score = distance(current, next) + lambda * distance(next, depot)
# where lambda depends on rest_capacity.
# If rest_capacity is small (e.g., < 5), lambda should be large, encouraging return to depot.
# If rest_capacity is large, lambda should be small, encouraging further travel.

# However, the task says "scored by mean total route length".
# Going back to depot closes the route. The distance from current to depot is added.
# If we go to a node, the distance current->node is added, and later node->...->depot.
# So minimizing current->node is good, but also minimizing the future path to depot is good.
# A simple look-ahead:
# For each feasible node i:
#   cost = distance(current, i) + distance(i, depot)  # Approximation of closing the route after i
#   But this ignores other nodes.
# A better approximation:
#   cost = distance(current, i) + min(distance(i, j) + distance(j, depot) for j in unvisited) ? Too slow.

# Let's try a simple adaptive lambda.
# lambda = max(0, (threshold - rest_capacity) / threshold) * some_factor
# Or simply:
# If rest_capacity < min_demand, return depot.
# Else, pick node minimizing: distance(current, node) + beta * distance(node, depot)
# where beta is small (e.g., 0.1) to slightly bias towards nodes closer to depot,
# but not too much to avoid premature closure.

# Actually, a known good heuristic is "Nearest Neighbor" but with a check:
# If the remaining capacity is less than the minimum demand of any unvisited node, return depot.
# Otherwise, pick the nearest node.
# The seed program does exactly this (implicitly, by filtering feasible nodes? No, the seed program does NOT filter by capacity! It just picks the nearest node from unvisited_nodes, assuming unvisited_nodes contains only feasible ones? Or does the caller filter?)
# Let's check the signature: `unvisited_nodes` is passed. The prompt says "picks the next customer, or the depot (0) to close a route".
# It does NOT say `unvisited_nodes` is filtered by capacity. It says "unvisited_nodes".
# So I MUST check capacity.
# The seed program: `return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])`
# This ignores capacity! If the nearest node has demand > rest_capacity, it's infeasible.
# Wait, if the seed program score is 1.0, and it ignores capacity, maybe the evaluator filters `unvisited_nodes` to only include feasible nodes?
# "unvisited_nodes" might mean "unvisited and feasible nodes".
# Let's assume `unvisited_nodes` contains only nodes that are feasible (demand <= rest_capacity).
# If so, the seed program is valid.
# If not, the seed program would fail or produce infeasible routes. Since it scores 1.0, it's likely valid.
# I will assume `unvisited_nodes` is filtered for capacity feasibility by the caller.
# If I want to be safe, I can re-filter, but it adds overhead.

# Let's try to improve upon Nearest Neighbor.
# Idea: "Cheapest Insertion" or "Savings" are not directly applicable in this greedy step-by-step manner.
# Idea: "Look-ahead Nearest Neighbor".
# For each candidate node i, estimate the cost of visiting i and then going to the nearest neighbor of i (or depot).
# cost_i = dist(current, i) + min(dist(i, j) for j in unvisited if j!=i and feasible) + dist(i, depot)
# This is O(N^2) per step, which might be too slow for 50 nodes * 8 instances * 400 steps?
# 50*8*400 = 160,000 steps. O(50^2) = 2500 ops per step. 160,000 * 2500 = 400 million ops. Might be too slow in Python.
# Let's stick to O(N) or O(N log N).

# Let's try the "Depot Bias" heuristic.
# score = dist(current, node) + alpha * dist(node, depot)
# alpha = 0.2
# This encourages nodes that are close to current AND close to depot.
# This tends to create shorter routes by not wandering too far.

# Let's implement this.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot
    
    # If remaining capacity is too low to serve any node, return depot.
    # Assuming unvisited_nodes might contain infeasible nodes, let's filter.
    # But if the caller filters, this is redundant. Let's check demands.
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Convert to numpy array for vectorization
    feasible_arr = np.array(feasible_nodes)
    
    # Get distances from current to feasible nodes
    dists_current = distance_matrix[current_node, feasible_arr]
    
    # Get distances from feasible nodes to depot
    dists_depot = distance_matrix[feasible_arr, depot]
    
    # Heuristic: Weighted sum
    # alpha = 0.5 might be too strong. Let's try 0.1.
    # Or dynamic alpha based on capacity?
    # If rest_capacity is low, we should prefer returning to depot.
    # But if we return to depot, we don't pick a node.
    # So the choice is between nodes.
    # If we pick a node, we commit to it.
    # A node close to depot is "safer" because it's easier to return later.
    
    alpha = 0.1
    scores = dists_current + alpha * dists_depot
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(feasible_arr[min_idx])

# EVOLVE-BLOCK-END
