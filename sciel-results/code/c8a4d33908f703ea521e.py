# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian relaxation with subgradient updates.
    
    Strategy:
    1. Dualize facility opening costs via Lagrangian multipliers.
    2. Solve relaxed subproblems: each customer picks cheapest facility considering multiplier.
    3. Repair infeasible capacity violations via local moves.
    4. Update multipliers using subgradient method.
    5. Track best feasible solution.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Initialize multipliers (penalties for capacity violations)
    multipliers = np.zeros(n_facilities)
    
    best_assignment = None
    best_cost = float('inf')
    
    def total_cost(assignment):
        """Compute total cost for a given assignment."""
        assign_total = np.sum(assign_cost[np.arange(n_customers), assignment])
        used = np.unique(assignment)
        open_total = np.sum(open_cost[used])
        return assign_total + open_total
    
    def repair(assignment, load):
        """Fix capacity violations by moving customers to feasible facilities."""
        assignment = assignment.copy()
        load = load.copy()
        
        for _ in range(10):  # Max repair iterations
            violated = np.where(load > capacities)[0]
            if len(violated) == 0:
                break
            
            for f in violated:
                # Find customers assigned to f
                custs = np.where(assignment == f)[0]
                # Sort by demand descending (move largest first)
                custs = custs[np.argsort(-demands[custs])]
                
                excess = load[f] - capacities[f]
                for i in custs:
                    if excess <= 0:
                        break
                    
                    # Find best feasible facility for customer i
                    feasible = np.where(load + demands[i] <= capacities)[0]
                    feasible = feasible[feasible != f]
                    if len(feasible) == 0:
                        # Try any facility with least load
                        feasible = np.where(load < capacities)[0]
                    
                    if len(feasible) > 0:
                        # Pick cheapest feasible
                        best_f = feasible[np.argmin(assign_cost[i, feasible])]
                        load[f] -= demands[i]
                        load[best_f] += demands[i]
                        assignment[i] = best_f
                        excess -= demands[i]
        
        return assignment, load
    
    # Number of subgradient iterations
    max_iter = 200
    step_size_base = 50.0
    best_dual = float('inf')
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Solve relaxed problem: each customer picks min(assign_cost[i,f] + multiplier[f])
        # But we need to respect capacity in a soft way - the multiplier penalizes overuse
        # Actually, for Lagrangian relaxation of capacity constraints:
        # L(lambda) = min sum_i sum_f x[i,f]*(assign_cost[i,f] + lambda[f]) + sum_f open_cost[f]*y[f]
        # where y[f] = 1 if any customer assigned to f
        # The subproblem decomposes: each customer independently picks min over f of (assign_cost[i,f] + lambda[f])
        # But this ignores opening costs and capacity. We handle capacity via repair.
        
        # For the relaxed problem, add multiplier to assignment cost
        relaxed_cost = assign_cost + multipliers[np.newaxis, :]
        
        # Each customer picks the facility with minimum relaxed cost
        assignment = np.argmin(relaxed_cost, axis=1)
        
        # Compute load
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        # Repair capacity violations
        assignment, load = repair(assignment, load)
        
        # Compute actual cost
        cost = total_cost(assignment)
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
        
        # Subgradient update for multipliers
        # Subgradient of L with respect to lambda[f] is (load[f] - capacity[f])
        # If load > capacity, increase lambda to discourage use
        # If load < capacity, decrease lambda (but keep non-negative)
        
        # Compute subgradient
        subgrad = load - capacities
        
        # Step size
        gap = best_cost - np.sum(assign_cost[np.arange(n_customers), assignment]) - np.sum(open_cost[np.unique(assignment)])
        # Use a simpler step size
        step_size = step_size_base / (it + 1)
        
        # Update multipliers
        multipliers += step_size * subgrad
        multipliers = np.maximum(multipliers, 0)  # Keep non-negative
    
    # Final local search on best assignment
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try single moves
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                
                current_f = assignment[i]
                # Try all other facilities
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    
                    # Check capacity
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Compute cost delta
                    delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                    
                    # Open cost change
                    count_current = np.sum(assignment == current_f)
                    count_f = np.sum(assignment == f)
                    
                    delta_open = 0.0
                    if count_current == 1:
                        delta_open -= open_cost[current_f]
                    if count_f == 0:
                        delta_open += open_cost[f]
                    
                    delta_total = delta_assign + delta_open
                    
                    if delta_total < -1e-9:
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        assignment[i] = f
                        best_cost += delta_total
                        best_assignment = assignment.copy()
                        improved = True
                        break
            
            # Try pair swaps
            if time.time() > deadline:
                break
                
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_customers):
                    if time.time() > deadline:
                        break
                    
                    fi = assignment[i]
                    fj = assignment[j]
                    
                    if fi == fj:
                        continue
                    
                    if load[fj] - demands[j] + demands[i] > capacities[fj]:
                        continue
                    if load[fi] - demands[i] + demands[j] > capacities[fi]:
                        continue
                    
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                    
                    if delta_assign < -1e-9:
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        assignment[i] = fj
                        assignment[j] = fi
                        best_cost += delta_assign
                        best_assignment = assignment.copy()
                        improved = True
                        break
                else:
                    continue
                break
    
    if best_assignment is None:
        # Fallback: greedy
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                    break
            else:
                assignment[i] = 0
                load[0] += demands[i]
        best_assignment = assignment
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
