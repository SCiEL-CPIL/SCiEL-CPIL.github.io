"""
Strategy: Iterated Local Search (ILS) with 2-Opt.
Instead of a greedy step-by-step heuristic, we treat the problem as a global optimization.
At each step (or periodically), we:
1. Complete the current partial tour using a fast greedy method (Nearest Neighbor).
2. Apply 2-Opt local search to minimize the total tour length.
3. Select the next node as the one that follows the current node in this optimized tour.

This ensures that each decision is informed by the global structure of the instance,
avoiding the "short-sightedness" of pure greedy heuristics.
We use a time budget to limit the number of ILS iterations per step, ensuring we finish
within the 150s limit. Since n=50 is small, 2-Opt is very fast, allowing many restarts.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global state
_tour_state = None
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin

def _reset_state():
    global _tour_state, _start_time
    _tour_state = None
    _start_time = time.time()

def _greedy_complete_tour(current_node, unvisited_nodes, distance_matrix):
    """
    Complete the tour from current_node visiting all unvisited nodes using Nearest Neighbor.
    Returns the full tour list starting from current_node.
    """
    if len(unvisited_nodes) == 0:
        return [current_node]
    
    tour = [current_node]
    remaining = list(unvisited_nodes)
    current = current_node
    
    # Pre-compute distances from current to all for efficiency? 
    # Since n=50, simple loop is fine.
    while remaining:
        # Find nearest neighbor in remaining
        # Vectorized: distance_matrix[current, remaining]
        dists = distance_matrix[current, remaining]
        idx = np.argmin(dists)
        next_node = remaining[idx]
        tour.append(next_node)
        current = next_node
        remaining.pop(idx)
        
    return tour

def _two_opt(tour, distance_matrix):
    """
    Apply 2-Opt local search to improve the tour.
    tour: list of nodes
    Returns improved tour.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    # Pre-compute distance matrix for the tour nodes for faster access?
    # distance_matrix is already global and fast (numpy array access).
    
    improved = True
    # Limit iterations to prevent infinite loops in case of numerical issues (though unlikely)
    max_iter = 10000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if adjacent (i+1 == j) or if it's the closing edge (i=1, j=n-1) which is handled by symmetry
                if j == i + 1:
                    continue
                
                # Calculate delta cost for reversing segment i+1..j
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                # If i=1, a is tour[0]. If j=n-1, d is tour[0].
                # The formula holds.
                
                d_old = distance_matrix[a, b] + distance_matrix[c, d]
                d_new = distance_matrix[a, c] + distance_matrix[b, d]
                
                if d_new < d_old - 1e-10:
                    # Reverse segment i..j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    # After a move, we can restart the inner loop or continue. 
                    # Restarting is safer but slower. For n=50, continuing is fine.
                    # However, to ensure convergence, restarting the outer loop is standard.
                    # But for speed in ILS, we often just continue scanning.
                    # Let's just break and restart the outer loop for correctness.
                    break
            if improved:
                # Restart outer loop to ensure no moves are missed due to the change
                break
                
    return tour

def _select_next_node_ils(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use ILS to find the best next node.
    """
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # 1. Greedy completion
    tour = _greedy_complete_tour(current_node, unvisited_nodes, distance_matrix)
    
    # 2. 2-Opt optimization
    # We can run multiple iterations with random perturbations (ILS) if time permits
    # But for a single step, a single 2-Opt pass is a good baseline.
    # To improve, we can do a few restarts with random perturbations if time allows.
    
    # For simplicity and speed per call, let's do one 2-Opt pass first.
    # If we have lots of time, we could do ILS.
    # Given n=50, 2-Opt is very fast. Let's do a few ILS iterations.
    
    best_tour = tour
    best_len = np.sum([distance_matrix[best_tour[i], best_tour[(i+1)%len(best_tour)]] for i in range(len(best_tour))])
    
    # Check time
    if _start_time is not None:
        elapsed = time.time() - _start_time
        # Allow up to 0.5s per step? 
        # Actually, we have 150s total. 
        # If we do this at every step (50 steps), we have 3s per step.
        # 2-Opt on 50 nodes is extremely fast (<1ms).
        # So we can afford many ILS iterations.
        
        # Let's run ILS for a fixed number of iterations or until time per step limit.
        # Time per step budget: 2.5 seconds (leaving 25s margin for overhead).
        step_start = time.time()
        time_budget_per_step = 2.5
        
        # ILS Loop
        current_tour = best_tour.copy()
        
        while (time.time() - step_start) < time_budget_per_step:
            # Perturb: Random 2-Opt move or 3-Opt move?
            # Simple perturbation: pick two random indices and swap/reverse
            n = len(current_tour)
            if n < 4:
                break
                
            i = random.randint(1, n - 2)
            j = random.randint(i + 1, n - 1)
            
            # Reverse segment i..j
            perturbed_tour = current_tour.copy()
            perturbed_tour[i:j+1] = reversed(perturbed_tour[i:j+1])
            
            # Local Search (2-Opt) on perturbed tour
            optimized_tour = _two_opt(perturbed_tour, distance_matrix)
            
            # Calculate length
            opt_len = np.sum([distance_matrix[optimized_tour[k], optimized_tour[(k+1)%n]] for k in range(n)])
            
            if opt_len < best_len - 1e-10:
                best_len = opt_len
                best_tour = optimized_tour
                current_tour = optimized_tour # Accept and continue from here
            else:
                # Simulated Annealing style? Or just reject?
                # For constructive heuristic, rejecting non-improving moves keeps us on the best found so far.
                # But ILS usually moves to the neighbor to escape local minima.
                # Let's move to the optimized perturbed tour even if it's worse, to explore?
                # No, for TSP, staying in the best basin is usually better unless we are stuck.
                # Let's just accept if better. If not, we can try to escape by forcing a worse move occasionally.
                # For now, simple hill climbing on the perturbed landscape.
                pass
            
    # Extract next node
    # The tour starts with current_node. The next node is best_tour[1].
    next_node = best_tour[1]
    return int(next_node)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _tour_state, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _tour_state = []
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        return next_node
    
    # Use ILS heuristic
    next_node = _select_next_node_ils(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    return next_node
# EVOLVE-BLOCK-END
