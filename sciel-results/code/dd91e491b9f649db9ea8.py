"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Hard deadline guard: return depot to close route if time is running out
    if time.time() - _start_time > 140:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Clarke-Wright Savings Heuristic
    # We want to maximize the "savings" of extending the current route to node i
    # instead of closing the route now (going to depot) and starting a new route to i later.
    
    # Let C_close be the cost if we close the route now:
    # We go from current_node to depot.
    # Cost = distance_matrix[current_node, depot]
    
    # Let C_extend_i be the cost if we go to i next, then eventually to depot:
    # We go from current_node to i, then from i to depot.
    # Cost = distance_matrix[current_node, i] + distance_matrix[i, depot]
    
    # The "saving" is not simply C_close - C_extend_i, because C_extend_i is an under-estimate
    # of the total cost if i is just one stop. However, in the context of the next decision,
    # the standard savings formula for merging two routes (depot->A... and depot->B...) is:
    # S = d(depot, A) + d(depot, B) - d(A, B)
    
    # Here, we are appending to an existing route.
    # If we close now: We pay d(current, depot). The node i will be visited in a future route starting from depot.
    # The "potential" cost of i in a future route is at least d(depot, i) + d(i, next_in_future_route).
    # This is hard to estimate without lookahead.
    
    # A more robust local heuristic for "should I extend or close" is based on the marginal cost.
    # If d(current, i) + d(i, depot) < d(current, depot) + d(depot, i) * factor?
    # Actually, let's look at the "Savings" of the edge (current, i).
    # In a standard CW algorithm, we precompute savings for all pairs (i, j) as d(0,i)+d(0,j)-d(i,j).
    # We maintain a list of routes. Here, we are building one route at a time (or extending the current one).
    
    # If we treat the current route as a "chain" ending at `current_node`, and we consider adding `i`:
    # The "saving" of connecting `current` to `i` instead of `current`->`depot` and `depot`->`i` is:
    # S_i = d(depot, current) + d(depot, i) - d(current, i)
    # Wait, d(depot, current) is fixed if we close now.
    # If we close now, cost incurred is d(current, depot).
    # If we go to i, cost incurred is d(current, i).
    # The "benefit" of going to i is that we avoid the trip depot->i later?
    # No, we will still have to go to i eventually.
    
    # Let's use a simpler, effective proxy for CW Savings in this sequential setting:
    # We prioritize nodes that are "close" to the current node relative to their distance to the depot.
    # Score = d(current, i) / (d(depot, i) + epsilon)
    # Minimizing this ratio favors nodes that are close to current AND far from depot?
    # No, if i is far from depot, d(depot, i) is large, ratio is small. Good.
    # If i is close to current, d(current, i) is small, ratio is small. Good.
    # This effectively maximizes d(depot, i) - d(current, i) roughly?
    
    # Let's try the standard CW saving formula applied to the "virtual" connection:
    # Saving(i) = d(depot, current) + d(depot, i) - d(current, i)
    # If Saving(i) is large, it means the triangle inequality benefit of connecting current->i is high
    # compared to going current->depot->i.
    # Specifically, if d(current, i) is small, Saving is large.
    # If d(depot, i) is large, Saving is large.
    
    # Let's compute this saving for all feasible nodes.
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Savings = d(current, depot) + d(depot, i) - d(current, i)
    savings = d_current_depot + d_depot_feasible - d_current_feasible
    
    # We want to MAXIMIZE savings.
    # However, we must ensure we don't pick a node that is very far away just because it has high savings
    # if the immediate distance is huge.
    # Actually, the savings formula inherently penalizes large d(current, i).
    # If d(current, i) is huge, savings becomes negative or small.
    
    # Let's verify:
    # If d(current, i) = 100, d(depot, i) = 10, d(current, depot) = 10.
    # Savings = 10 + 10 - 100 = -80. Bad.
    # If d(current, i) = 5, d(depot, i) = 10, d(current, depot) = 10.
    # Savings = 10 + 10 - 5 = 15. Good.
    
    # So maximizing savings is a sound heuristic.
    
    best_idx = np.argmax(savings)
    
    # Safety check: if all savings are very negative, it might be better to close the route?
    # The problem statement says "return 0 to close a route".
    # If the best saving is negative, it means going to any node increases the cost compared to closing now
    # AND starting a new route to that node (ignoring the cost of the new route's start).
    # Actually, if we close now, we pay d(current, depot).
    # If we go to i, we pay d(current, i).
    # The "cost difference" is d(current, i) - d(current, depot).
    # If d(current, i) > d(current, depot), we are paying more to go to i than to go home.
    # BUT, we save the trip depot->i later?
    # The savings formula accounts for this: Saving = d(depot, current) + d(depot, i) - d(current, i).
    # If Saving < 0, then d(current, i) > d(depot, current) + d(depot, i).
    # This implies the triangle inequality is violated significantly or i is very far.
    # In Euclidean space, d(current, i) <= d(current, depot) + d(depot, i) always holds.
    # So Savings is ALWAYS >= 0 in Euclidean space.
    # Therefore, we should almost always extend if there is a feasible node, unless capacity is tight?
    # Wait, the heuristic is about TOTAL route length.
    # If we always extend, we might end up with one giant route that is inefficient due to capacity constraints forcing early termination?
    # No, we check capacity.
    
    # However, there is a nuance: if we go to i, we might be forced to go to depot sooner than optimal
    # because of capacity.
    # A simple CW heuristic doesn't look ahead to capacity.
    # But for N=50, capacity=50, demands 1-9, average demand ~5.
    # Max load ~250. Capacity 50 means ~5 stops per route.
    # So routes are short.
    
    # Let's stick to the Savings heuristic. It is proven to be strong for CVRP.
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
