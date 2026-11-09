import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(d):
    """
    Constructs a tour using the Nearest Neighbor heuristic starting from node 0.
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Get distances from current to all unvisited nodes
        # Mask visited nodes with inf
        dists = d[current].copy()
        dists[visited] = np.inf
        # The current node itself is visited, so it's inf. 
        # If n==1, this would be a problem, but n=50.
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _two_opt_optimize(tour, d):
    """
    Optimizes a tour using 2-opt.
    Returns the optimized tour and its length.
    """
    n = len(tour) - 1 # tour has n+1 elements, last is first
    if n <= 2:
        length = sum(d[tour[i], tour[i+1]] for i in range(n))
        return tour, length
    
    def get_length(t):
        return sum(d[t[i], t[i+1]] for i in range(len(t)-1))
    
    improved = True
    current_length = get_length(tour)
    
    while improved:
        improved = False
        # Standard 2-opt: remove edges (i-1, i) and (j, j+1) and replace with (i-1, j) and (i, j+1)
        # Indices in tour: 0..n
        # Edges are (tour[k], tour[k+1]) for k in 0..n-1
        # Reversing segment tour[i..j]
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edge 1: (tour[i-1], tour[i])
                # Edge 2: (tour[j], tour[j+1])
                # New Edge 1: (tour[i-1], tour[j])
                # New Edge 2: (tour[i], tour[j+1])
                
                old_len = d[tour[i-1], tour[i]] + d[tour[j], tour[j+1]]
                new_len = d[tour[i-1], tour[j]] + d[tour[i], tour[j+1]]
                
                if new_len < old_len - 1e-10:
                    # Reverse the segment tour[i..j]
                    tour[i:j+1] = tour[i:j+1][::-1]
                    current_length = new_len + (current_length - old_len)
                    improved = True
                    # Early exit from inner loop is possible but full pass is safer for convergence
                    # Actually, just continue to check other pairs, or restart? 
                    # Standard 2-opt usually does a full pass then repeats.
                    # For speed, we can just break inner loop and restart outer? 
                    # No, just continue. The change might affect other pairs.
                    # But for n=50, O(N^2) is 2500 ops. Very fast.
                    
    return tour, current_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin for 300s limit
    
    d = distance_matrix
    n = d.shape[0]
    
    # 1. Generate a high-quality baseline tour
    # Nearest Neighbor is O(N^2)
    nn_tour = _nearest_neighbor_tour(d)
    
    # 2. Optimize with 2-Opt
    # 2-Opt is O(N^2) per pass, usually converges in a few passes
    optimized_tour, _ = _two_opt_optimize(nn_tour, d)
    
    # Check time
    if time.time() > deadline - 50.0:
        # Fallback to simple inverse distance if we are running out of time
        # Though NN+2Opt for N=50 is extremely fast (<1ms usually)
        return 1.0 / (d + 1e-9)
    
    # 3. Construct Heuristic Matrix
    # Strategy: 
    # Base: Inverse distance (standard ACO heuristic)
    # Boost: Edges in the optimized tour get a massive multiplier.
    # This guides the ants to follow the known good tour structure, 
    # allowing pheromones to reinforce the correct edges quickly, 
    # while the inverse distance base allows exploration of alternatives 
    # if the 2-opt local optimum is not the global one (though for 50 points it's likely very good).
    
    # Base heuristic: 1/d
    h_base = 1.0 / (d + 1e-9)
    
    # Create the final heuristic
    h_final = h_base.copy()
    
    # Identify edges in the optimized tour
    # The tour is closed: 0 -> ... -> 0
    # Edges are (tour[i], tour[i+1]) for i in 0..n-1
    # Note: The tour list from _two_opt_optimize is [0, ..., 0]
    # So tour[0]=0, tour[n]=0
    
    # Boost factor: 
    # If we set h[edge] to a very large number, ants will almost always pick these edges.
    # This essentially turns the ACO into a verification/refinement of the 2-opt tour.
    # However, if the 2-opt tour is perfect, this is great. 
    # If not, the ants might get stuck. 
    # A moderate boost allows deviation.
    # Let's try a high boost, e.g., 100x or 1000x.
    # Since pheromone starts at 1, and heuristic is multiplied, 
    # if h is 100, probability is 100x higher than average edges (where h~1).
    
    boost_factor = 100.0
    
    for i in range(n):
        u = optimized_tour[i]
        v = optimized_tour[i+1]
        # Only boost forward direction? 
        # The task says "pheromone[i, j] for each step i to j ... in travel direction only".
        # The heuristic matrix is used as heuristic[i, j].
        # Ants move i -> j. So we boost h[u, v].
        # Should we boost h[v, u]? 
        # The tour is closed, so the reverse direction is not taken in the tour.
        # However, TSP is symmetric usually? 
        # "50 uniform random points in the unit square" implies Euclidean distances, which are symmetric.
        # The distance matrix d is symmetric.
        # The ACO description says "adds 1/length to pheromone[i, j] ... in travel direction only".
        # This implies the pheromone matrix becomes asymmetric over time.
        # The heuristic is computed once. 
        # If we only boost h[u, v], ants going u->v are strongly guided.
        # If they need to go v->u (which they won't in the optimized tour), they aren't guided.
        # But since we want them to follow the tour, boosting the forward direction is correct.
        
        h_final[u, v] *= boost_factor
        
    # Optional: Also boost the reverse direction? 
    # If the heuristic is symmetric, it might help if the ant gets "lost" and needs to backtrack?
    # But ants don't backtrack, they must visit all nodes.
    # Let's stick to forward direction to match the tour structure exactly.
    
    # Safety check: ensure no NaNs or Inf
    if not np.all(np.isfinite(h_final)):
        h_final = np.nan_to_num(h_final, nan=1e-9, posinf=1e9, neginf=1e-9)
        
    # Ensure positive
    h_final = np.maximum(h_final, 1e-9)
    
    return h_final
# EVOLVE-BLOCK-END
