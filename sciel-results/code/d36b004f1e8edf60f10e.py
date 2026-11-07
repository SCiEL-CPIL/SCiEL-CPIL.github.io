"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a Savings-Enhanced Distance Model.
    
    The heuristic is defined as:
    eta_ij = (max(0, savings_ij))^gamma_s * (1 / d_ij)^gamma_d
    
    where savings_ij = d_i0 + d_0j - d_ij.
    
    We tune gamma_s and gamma_d using a fast Greedy Savings proxy to find 
    parameters that lead to low-cost routes.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Precompute savings matrix
    # savings[i, j] = dm[i, 0] + dm[0, j] - dm[i, j]
    # Note: In CVRP, we usually consider i != 0 and j != 0 for customer-customer edges.
    # For depot edges, the logic is slightly different, but the ACO handles transitions.
    # The heuristic matrix needs to be defined for all i, j.
    # If i=0 or j=0, the "savings" concept of avoiding a depot return is less direct.
    # However, a common heuristic for ACO in VRP is just 1/d.
    # Let's apply the savings logic primarily to customer-customer edges and 
    # fall back to a distance-based heuristic for depot edges, or simply use 
    # the savings value which naturally becomes 0 or negative for some depot edges 
    # if not careful, but actually:
    # If i=0, savings[0, j] = d_00 + d_0j - d_0j = 0. This would make depot edges unattractive if gamma_s > 0.
    # This is a problem. Ants MUST leave the depot.
    
    # Revised Strategy:
    # Use the savings-based heuristic ONLY for customer-to-customer transitions (i>0, j>0, i!=j).
    # For transitions involving the depot (i=0 or j=0), use a standard distance heuristic 1/d^gamma_d.
    # This respects the structure: "Savings" matter when linking two customers to avoid a depot trip.
    # Leaving/entering depot is a necessary step, so we just prefer shorter distances.
    
    depot_dists = dm[0, :]
    
    def compute_heuristic(gamma_s, gamma_d):
        """
        Compute the heuristic matrix for given gamma_s and gamma_d.
        """
        eta = np.zeros((n, n))
        
        # 1. Customer-Customer edges (i > 0, j > 0, i != j)
        # Savings S_ij = d_i0 + d_0j - d_ij
        # We want S_ij > 0 for it to be "good".
        # Matrix of d_i0: dm[:, 0]
        # Matrix of d_0j: dm[0, :]
        # Matrix of d_ij: dm
        
        # Broadcasting:
        # d_i0 is shape (n, 1)
        # d_0j is shape (1, n)
        # dm is shape (n, n)
        
        d_i0 = dm[:, 0:1]
        d_0j = dm[0:1, :]
        
        savings = d_i0 + d_0j - dm
        
        # Clip negative savings to 0
        savings = np.maximum(savings, 0.0)
        
        # Distance term
        d_safe = np.maximum(dm, 1e-9)
        dist_term = 1.0 / np.power(d_safe, gamma_d)
        
        # Savings term
        # Avoid 0^0 issues, but savings can be 0. If gamma_s > 0, 0^gamma_s = 0.
        # If savings is 0, the edge is not "saving" anything, so attractiveness from savings is 0.
        # We add a small epsilon to savings to avoid log issues if we used logs, but here it's power.
        # If savings is 0, term is 0.
        sav_term = np.power(savings + 1e-9, gamma_s)
        
        # Combined for customer-customer
        eta_cc = sav_term * dist_term
        
        # Apply only to customer-customer part (indices 1..n-1)
        eta[1:, 1:] = eta_cc[1:, 1:]
        
        # 2. Depot-Customer edges (i=0, j>0 or i>0, j=0)
        # Use pure distance heuristic
        # For i=0, j>0: eta[0, j] = dist_term[0, j]
        # For i>0, j=0: eta[i, 0] = dist_term[i, 0]
        
        # dist_term is already computed for all i,j
        eta[0, 1:] = dist_term[0, 1:]
        eta[1:, 0] = dist_term[1:, 0]
        
        # 3. Diagonal (i=j) should be very low or 0 to prevent self-loops
        # The ACO logic usually skips visited nodes, but diagonal entries in the matrix
        # might be accessed if not careful. Set to 1e-9.
        np.fill_diagonal(eta, 1e-9)
        
        # Ensure no negative or non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        return eta

    def greedy_proxy(eta, n_ants=5):
        """
        Fast deterministic proxy: Greedy construction based on the heuristic.
        Instead of running full ACO, we run a few greedy constructions 
        that pick the next node with the highest eta value among feasible ones.
        This is much faster than ACO and correlates well with heuristic quality.
        """
        best_cost = np.inf
        rng = np.random.default_rng(42) # Fixed seed for proxy stability
        
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = set()
            route_cost = 0.0
            
            while True:
                # Find feasible unvisited customers
                feasible = []
                for j in range(1, n):
                    if j not in visited and dem[j] <= cap - load:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    route_cost += dm[current, 0]
                    break
                
                # Pick the feasible node with the highest heuristic value
                # If we want to introduce some stochasticity to avoid getting stuck in 
                # a very bad local greedy trap, we could sample from top-k, 
                # but for a proxy, pure greedy is fast and robust enough for ranking.
                # Let's add a small random perturbation to break ties or escape plateaus.
                # Actually, let's just pick the max.
                
                max_eta = -1
                next_node = -1
                for j in feasible:
                    val = eta[current, j]
                    if val > max_eta:
                        max_eta = val
                        next_node = j
                
                if next_node == -1:
                    # Fallback if something went wrong
                    break
                    
                route_cost += dm[current, next_node]
                load += dem[next_node]
                visited.add(next_node)
                current = next_node
                
            if route_cost < best_cost:
                best_cost = route_cost
                
        return best_cost

    # Search for optimal gamma_s and gamma_d
    # gamma_s: weight of savings (0 to 2)
    # gamma_d: weight of distance (0 to 3)
    
    # Grid search
    gamma_s_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    gamma_d_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    best_gs = 1.0
    best_gd = 1.0
    best_score = np.inf
    
    for gs in gamma_s_vals:
        if time.time() > deadline:
            break
        for gd in gamma_d_vals:
            if time.time() > deadline:
                break
            
            eta = compute_heuristic(gs, gd)
            score = greedy_proxy(eta, n_ants=5)
            
            if score < best_score:
                best_score = score
                best_gs = gs
                best_gd = gd
                
    # If the grid search didn't find anything (shouldn't happen), fallback to standard 1/d
    if best_score == np.inf:
        best_gs = 0.0
        best_gd = 1.0
        
    # Generate final heuristic
    final_heuristic = compute_heuristic(best_gs, best_gd)
    
    # Final safety checks
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    final_heuristic[final_heuristic <= 0] = 1e-9
    
    return final_heuristic
# EVOLVE-BLOCK-END
