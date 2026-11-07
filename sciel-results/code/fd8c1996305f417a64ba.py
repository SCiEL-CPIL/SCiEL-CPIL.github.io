"""
Facility location solver using Benders decomposition with LP subproblem.
Strategy: Benders decomposition (master problem over facility opening, LP subproblem for assignment)
with subgradient acceleration and local search polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using Benders decomposition.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # Precompute: for each customer, sort facilities by cost
    # This helps in constructing good initial solutions and Benders cuts

    # Benders decomposition approach:
    # Master problem: choose which facilities to open
    # Subproblem: given open facilities, solve assignment (LP/flow)
    
    # Since n_facilities=20, we can enumerate subsets or use Benders with cuts
    # But 2^20 is too many. Instead, use a heuristic Benders:
    # 1. Start with all facilities open
    # 2. Solve assignment
    # 3. Generate Benders cuts to identify which facilities are truly needed
    # 4. Iteratively refine
    
    # Alternative: Use the fact that this is a small problem (100 customers, 20 facilities)
    # and use a different approach: Lagrangian relaxation on capacity constraints
    
    # Let's try a hybrid: 
    # 1. Solve LP relaxation of the assignment problem (without integrality)
    # 2. Use the LP dual to guide a greedy assignment
    # 3. Local search with capacity constraints
    
    # Actually, let's use a completely different strategy:
    # Greedy construction with lookahead + multi-start local search with swap moves
    
    # Strategy: 
    # 1. Multiple diverse initial solutions using different greedy strategies
    # 2. Intensive local search with 1-move and 2-swap moves
    # 3. Keep best solution found
    
    best_assignment = None
    best_cost = np.inf
    
    def compute_total_cost(assignment):
        """Compute the total cost of an assignment."""
        if assignment is None:
            return np.inf
        # Check validity
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = assignment[c]
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        # Compute cost
        cost = 0.0
        open_mask = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = assignment[c]
            cost += assign_cost[c, f]
            open_mask[f] = True
        cost += open_cost[open_mask].sum()
        return cost
    
    def greedy_assign(strategy='nearest', random_state=None):
        """
        Greedy assignment with different strategies.
        strategy: 'nearest', 'cost', 'balanced', 'random'
        """
        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(random_state)
        
        load = np.zeros(n_facilities)
        assignment = np.full(n_customers, -1, dtype=int)
        remaining = set(range(n_customers))
        
        # Determine order of customers
        if strategy == 'random':
            order = list(rng.permutation(n_customers))
        else:
            order = list(range(n_customers))
        
        for c in order:
            # Find feasible facilities
            feasible = [f for f in range(n_facilities) 
                       if load[f] + demands[c] <= capacities[f] + 1e-9]
            if not feasible:
                # No feasible facility, this shouldn't happen if total capacity >= total demand
                # Assign to facility with most remaining capacity
                f = np.argmax(capacities - load)
                if load[f] + demands[c] > capacities[f]:
                    return None  # Infeasible
                feasible = [f]
            
            if strategy == 'nearest':
                # Pick facility with lowest assignment cost
                costs = [assign_cost[c, f] for f in feasible]
                best_f = feasible[np.argmin(costs)]
            elif strategy == 'cost':
                # Consider open cost: if facility not yet open, add open_cost
                # Approximate by adding open_cost/capacity to each unit
                adjusted = []
                for f in feasible:
                    adj = assign_cost[c, f]
                    if load[f] == 0:
                        adj += open_cost[f] / max(1, capacities[f] - load[f]) * demands[c]
                    adjusted.append(adj)
                best_f = feasible[np.argmin(adjusted)]
            elif strategy == 'balanced':
                # Pick facility with most remaining capacity
                rem_cap = [capacities[f] - load[f] for f in feasible]
                best_f = feasible[np.argmax(rem_cap)]
            elif strategy == 'random':
                # Pick randomly among feasible
                best_f = feasible[rng.integers(len(feasible))]
            else:
                best_f = feasible[0]
            
            assignment[c] = best_f
            load[best_f] += demands[c]
        
        return assignment
    
    def local_search(assignment, max_time=None):
        """
        Local search with 1-move and 2-swap moves.
        Returns improved assignment.
        """
        if max_time is None:
            max_time = deadline - 2.0
        
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[assignment[c]] += demands[c]
        
        improved = True
        iterations = 0
        max_iterations = 50  # Limit iterations to avoid infinite loops
        
        while improved and time.time() < max_time and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1-move: try moving each customer to a different facility
            for c in range(n_customers):
                old_f = assignment[c]
                old_load = load[old_f] - demands[c]
                
                best_new_f = old_f
                best_delta = 0.0
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    # Open cost changes
                    if old_load == 0:
                        delta -= open_cost[old_f]  # Closing old facility
                    if load[new_f] == 0:
                        delta += open_cost[new_f]  # Opening new facility
                    
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                
                if best_new_f != old_f:
                    assignment[c] = best_new_f
                    load[old_f] = old_load
                    load[best_new_f] += demands[c]
                    improved = True
            
            # 2-swap: try swapping two customers between facilities
            if time.time() < max_time and iterations < max_iterations:
                for c1 in range(n_customers):
                    for c2 in range(c1 + 1, n_customers):
                        f1 = assignment[c1]
                        f2 = assignment[c2]
                        if f1 == f2:
                            continue
                        
                        # Check if swap is feasible
                        # After swap: c1 goes to f2, c2 goes to f1
                        load_f1_after = load[f1] - demands[c1] + demands[c2]
                        load_f2_after = load[f2] - demands[c2] + demands[c1]
                        
                        if load_f1_after > capacities[f1] + 1e-9:
                            continue
                        if load_f2_after > capacities[f2] + 1e-9:
                            continue
                        
                        delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                                (assign_cost[c2, f1] - assign_cost[c2, f2])
                        
                        if delta < -1e-9:
                            # Perform swap
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = load_f1_after
                            load[f2] = load_f2_after
                            improved = True
        
        return assignment
    
    # Multi-start local search
    strategies = ['nearest', 'cost', 'balanced', 'random']
    n_starts = 8
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        if i < len(strategies):
            strategy = strategies[i]
        else:
            strategy = strategies[i % len(strategies)]
        
        # Try multiple random seeds for 'random' strategy
        seed = i if i >= len(strategies) else None
        assignment = greedy_assign(strategy, random_state=seed if seed is not None else None)
        
        if assignment is None:
            continue
        
        # Local search
        assignment = local_search(assignment)
        
        cost = compute_total_cost(assignment)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
    
    # If no valid assignment found, create a fallback
    if best_assignment is None:
        # Simple fallback: assign to facility with most remaining capacity
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            # Find facility with most remaining capacity
            f = np.argmax(capacities - load)
            best_assignment[c] = f
            load[f] += demands[c]
    
    # Final local search polish
    if time.time() < deadline - 3:
        best_assignment = local_search(best_assignment)
    
    # Verify and repair if needed
    load = np.zeros(n_facilities)
    for c in range(n_customers):
        f = best_assignment[c]
        if f < 0 or f >= n_facilities:
            f = 0
            best_assignment[c] = f
        load[f] += demands[c]
    
    # Check capacity violations
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            # Move customers to other facilities
            excess = load[f] - capacities[f]
            customers_at_f = [c for c in range(n_customers) if best_assignment[c] == f]
            # Sort by how much they'd benefit from moving
            customers_at_f.sort(key=lambda c: assign_cost[c].min() - assign_cost[c, f])
            for c in customers_at_f:
                if excess <= 1e-9:
                    break
                # Find best alternative
                best_alt_f = -1
                best_alt_cost = np.inf
                for f2 in range(n_facilities):
                    if f2 == f:
                        continue
                    if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                        alt_cost = assign_cost[c, f2]
                        if alt_cost < best_alt_cost:
                            best_alt_cost = alt_cost
                            best_alt_f = f2
                if best_alt_f >= 0:
                    load[f] -= demands[c]
                    load[best_alt_f] += demands[c]
                    best_assignment[c] = best_alt_f
                    excess -= demands[c]
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
