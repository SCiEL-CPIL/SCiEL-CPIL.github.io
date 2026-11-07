"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import gmres
import sys


def _solve_lp_facility_subset(open_cost, assign_cost, demands, capacities, active_facilities, time_limit=10.0):
    """
    Solve a relaxation where we decide which facilities to open and assign customers.
    We use a simple approach: for a given set of active facilities, solve the assignment LP.
    But we need to decide which facilities to open too.
    
    Actually, let's use a different approach:
    1. Start with all facilities open.
    2. Solve assignment LP.
    3. Identify facilities with low utilization or high cost.
    4. Try removing some facilities and re-solve.
    
    Or better: use Lagrangian relaxation or a greedy heuristic.
    """
    pass


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Initial greedy solution: largest demand first, cheapest feasible facility
    def greedy_solution():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Find cheapest feasible facility
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                # No feasible facility, assign to facility with most remaining capacity
                f = np.argmax(capacities - load)
            else:
                # Among feasible, pick cheapest
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            load[f] += demands[i]
            result[i] = f
        # Verify capacity constraints
        for f in range(n_facilities):
            total_load = np.sum(demands[result == f])
            if total_load > capacities[f] + 1e-6:
                return None  # Infeasible
        return result
    
    best_result = greedy_solution()
    if best_result is None:
        # Fallback: simple assignment ignoring capacity (shouldn't happen)
        best_result = np.argmin(assign_cost, axis=1).astype(int)
    
    def compute_score(result):
        if result is None:
            return float('inf')
        # Check validity
        if np.any(result < 0) or np.any(result >= n_facilities):
            return float('inf')
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[result[i]] += demands[i]
        if np.any(load > capacities + 1e-6):
            return float('inf')
        # Compute cost
        open_facilities = np.unique(result)
        cost = np.sum(open_cost[open_facilities])
        for i in range(n_customers):
            cost += assign_cost[i, result[i]]
        return cost
    
    best_score = compute_score(best_result)
    
    # Local search: try moving customers to different facilities
    def local_search(result, max_iterations=5000):
        result = result.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[result[i]] += demands[i]
        
        current_score = compute_score(result)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            # Try moving each customer to a better facility
            # Sort customers by potential improvement
            for i in range(n_customers):
                current_f = result[i]
                # Try all other facilities
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    # Check capacity
                    if load[f] + demands[i] > capacities[f] + 1e-6:
                        continue
                    # Compute delta
                    old_cost = assign_cost[i, current_f]
                    new_cost = assign_cost[i, f]
                    # Open cost change
                    delta_open = 0
                    if load[current_f] == demands[i]:  # Current facility becomes empty
                        delta_open -= open_cost[current_f]
                    if load[f] == 0:  # New facility becomes open
                        delta_open += open_cost[f]
                    delta = new_cost - old_cost + delta_open
                    if delta < -1e-9:
                        # Make the move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        current_score += delta
                        improved = True
                        break
        return result, current_score
    
    # Run local search
    improved_result, improved_score = local_search(best_result)
    if improved_score < best_score:
        best_result = improved_result
        best_score = improved_score
    
    # Try a different initial solution: smallest demand first
    def greedy_solution_small_first():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(demands)
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            load[f] += demands[i]
            result[i] = f
        for f in range(n_facilities):
            total_load = np.sum(demands[result == f])
            if total_load > capacities[f] + 1e-6:
                return None
        return result
    
    alt_result = greedy_solution_small_first()
    if alt_result is not None:
        alt_score = compute_score(alt_result)
        if alt_score < best_score:
            best_result = alt_result
            best_score = alt_score
            improved_result, improved_score = local_search(best_result)
            if improved_score < best_score:
                best_result = improved_result
                best_score = improved_score
    
    # Try multiple random restarts with greedy + local search
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Perturb the best solution
        perturbed = best_result.copy()
        # Randomly reassign some customers
        n_perturb = max(1, n_customers // 10)
        indices = rng.choice(n_customers, size=n_perturb, replace=False)
        for i in indices:
            feasible = np.where(np.zeros(n_facilities))[0]  # All facilities
            # Pick randomly from feasible
            f = rng.randint(n_facilities)
            perturbed[i] = f
        
        # Check if perturbed is feasible, if not, fix it
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[perturbed[i]] += demands[i]
        
        # Fix infeasibilities
        feasible = True
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-6:
                feasible = False
                # Move some customers from f to other facilities
                customers_at_f = np.where(perturbed == f)[0]
                for i in customers_at_f:
                    # Find a facility with available capacity
                    moved = False
                    for g in range(n_facilities):
                        if g == f:
                            continue
                        if load[g] + demands[i] <= capacities[g] + 1e-6:
                            load[f] -= demands[i]
                            load[g] += demands[i]
                            perturbed[i] = g
                            moved = True
                            break
                    if not moved:
                        # Can't fix, revert
                        load[f] += demands[i]
                        perturbed[i] = f
                        break
                if not feasible:
                    break
        
        if not feasible:
            continue
        
        # Run local search on perturbed solution
        ls_result, ls_score = local_search(perturbed, max_iterations=2000)
        if ls_score < best_score:
            best_result = ls_result
            best_score = ls_score
    
    # Final check
    final_score = compute_score(best_result)
    if final_score != float('inf'):
        return best_result.astype(int)
    else:
        # Fallback
        return np.argmin(assign_cost, axis=1).astype(int)

# EVOLVE-BLOCK-END
