"""Diverse randomized greedy seeds + fast vectorized 2-opt local search + SA for CFLP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: Multiple diverse randomized greedy initializations, followed by
    fast vectorized first-order (2-opt) local search with occasional 3-opt,
    and a short simulated annealing phase. No dual/Lagrangian guidance.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # Precompute for speed
    total_demand = demands.sum()

    def compute_cost(assignment):
        """Compute total cost, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def greedy_init(rng, diversity=0.0):
        """
        Greedy assignment with optional randomization for diversity.
        diversity in [0,1]: 0 = deterministic best-cost, 1 = fully random among feasible.
        """
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers: largest demand first, but with some randomness for diversity
        if diversity > 0.1:
            # Perturb the order
            base_order = np.argsort(-demands)
            # Shuffle a portion
            n_shuf = int(n_customers * diversity * 0.5)
            if n_shuf > 0:
                idx = rng.sample(range(n_customers), n_shuf)
                # Randomize positions of these customers
                for i in idx:
                    new_pos = rng.randint(0, n_customers - 1)
                    # Simple: just swap in the order array
                    tmp = base_order[i]
                    base_order[i] = base_order[new_pos]
                    base_order[new_pos] = tmp
            order = base_order
        else:
            order = np.argsort(-demands)
        
        for c in order:
            # Cost to assign c to each facility
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            
            if np.all(np.isinf(costs)):
                # No feasible: pick the one with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
            else:
                if diversity > 0.1:
                    # Softmin-like randomization: pick from top-k with probability
                    finite_mask = np.isfinite(costs)
                    finite_costs = costs[finite_mask]
                    finite_indices = np.where(finite_mask)[0]
                    
                    if len(finite_indices) == 0:
                        f = int(np.argmax(capacities - load))
                    else:
                        # Add noise to costs
                        noise = rng.exponential(scale=diversity * 5.0, size=len(finite_costs))
                        perturbed = finite_costs + noise
                        # Pick the minimum perturbed cost
                        best_local = int(np.argmin(perturbed))
                        f = int(finite_indices[best_local])
                else:
                    f = int(np.argmin(costs))
            
            result[c] = f
            load[f] += demands[c]
        
        return result

    def vectorized_local_search(assignment, max_passes=50):
        """
        Fast vectorized 2-opt local search.
        For each customer, try moving to any feasible facility with lower marginal cost.
        Uses vectorized operations where possible.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        
        best_cost = current_cost
        best_assign = assignment.copy()
        
        # Precompute: for each customer, the marginal cost of moving to each facility
        # marginal[c, f] = assign_cost[c, f] - assign_cost[c, assignment[c]]
        # + open_cost[f] if load[f] == 0 (newly opened)
        # - open_cost[assignment[c]] if load[assignment[c]] == demands[c] (closes)
        
        for _pass in range(max_passes):
            if time.time() > deadline - 15:
                break
            
            improved = False
            
            # Vectorized: compute best alternative for each customer
            # For each customer c, current facility is assignment[c]
            # We want to find f != assignment[c] such that:
            #   load[f] + demands[c] <= capacities[f]
            #   and delta_cost(c, f) < 0
            # where delta_cost = assign_cost[c,f] - assign_cost[c,assignment[c]]
            #   + open_cost[f] * (load[f] == 0)
            #   - open_cost[assignment[c]] * (load[assignment[c]] == demands[c])
            
            # Build cost matrix with adjustments
            # Base assignment cost difference
            current_f = assignment.astype(int)
            base_cost = assign_cost[np.arange(n_customers), current_f]
            
            # For each customer, compute delta to all facilities
            # delta[c, f] = assign_cost[c, f] - base_cost[c]
            delta = assign_cost - base_cost[:, None]
            
            # Penalty for opening new facility
            # If load[f] == 0, add open_cost[f]
            open_penalty = np.zeros((n_customers, n_facilities))
            zero_load_mask = (load == 0)  # shape (n_facilities,)
            if np.any(zero_load_mask):
                # For facilities with zero load, add open_cost
                open_penalty[:, zero_load_mask] = open_cost[zero_load_mask][None, :]
            
            # Reward for closing current facility
            # If load[current_f[c]] == demands[c], subtract open_cost[current_f[c]]
            # This is per-customer, so we handle it in the delta
            close_reward = np.zeros(n_customers)
            for c in range(n_customers):
                cf = int(current_f[c])
                if load[cf] == demands[c]:
                    close_reward[c] = open_cost[cf]
            
            # Total delta
            total_delta = delta + open_penalty - close_reward[:, None]
            
            # Mask out current facility
            for c in range(n_customers):
                total_delta[c, int(current_f[c])] = np.inf
            
            # Mask out infeasible (capacity)
            # feasible[c, f] = load[f] + demands[c] <= capacities[f]
            feasible = (load[None, :] + demands[:, None]) <= capacities[None, :]
            total_delta[~feasible] = np.inf
            
            # Find best move for each customer
            best_new_f = np.argmin(total_delta, axis=1)
            best_delta = total_delta[np.arange(n_customers), best_new_f]
            
            # Only consider moves with negative delta
            improve_mask = best_delta < -1e-9
            
            if not np.any(improve_mask):
                break
            
            # Apply all improving moves simultaneously (greedy)
            # But we need to update load carefully
            # Process in order of most negative delta first
            improve_indices = np.where(improve_mask)[0]
            improve_indices = improve_indices[np.argsort(best_delta[improve_indices])]
            
            for c in improve_indices:
                old_f = int(current_f[c])
                new_f = int(best_new_f[c])
                
                if new_f == old_f:
                    continue
                
                # Check capacity again (might have changed due to previous moves in this pass)
                if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                    continue
                
                # Compute actual delta with current load
                delta_actual = assign_cost[c, new_f] - assign_cost[c, old_f]
                if load[new_f] == 0:
                    delta_actual += open_cost[new_f]
                if load[old_f] == demands[c]:
                    delta_actual -= open_cost[old_f]
                
                if delta_actual < -1e-9:
                    load[old_f] -= demands[c]
                    load[new_f] += demands[c]
                    assignment[c] = new_f
                    current_cost += delta_actual
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assign = assignment.copy()
            
            if not improved:
                break
        
        return best_assign

    def simulated_annealing(assignment, max_iter=5000):
        """
        Short SA phase to escape local optima.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        
        best_cost = current_cost
        best_assign = assignment.copy()
        
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random move: pick a customer and a random feasible facility
            c = rng.randint(0, n_customers - 1)
            old_f = int(assignment[c])
            
            # Find feasible facilities
            feasible = (load + demands[c]) <= capacities
            feasible[old_f] = False  # Don't stay
            feasible_indices = np.where(feasible)[0]
            
            if len(feasible_indices) == 0:
                continue
            
            new_f = int(rng.choice(feasible_indices))
            
            # Compute delta
            delta = assign_cost[c, new_f] - assign_cost[c, old_f]
            if load[new_f] == 0:
                delta += open_cost[new_f]
            if load[old_f] == demands[c]:
                delta -= open_cost[old_f]
            
            # Accept or reject
            if delta < 0 or rng.random() < np.exp(-delta / T):
                load[old_f] -= demands[c]
                load[new_f] += demands[c]
                assignment[c] = new_f
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assign = assignment.copy()
            
            T *= cooling
        
        return best_assign

    # === Main algorithm ===
    best_solution = None
    best_cost = np.inf
    
    # Phase 1: Multiple diverse greedy initializations + local search
    n_seeds = 8
    for seed_idx in range(n_seeds):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(seed_idx * 12345 + 67890)
        diversity = 0.0 if seed_idx == 0 else min(0.8, 0.1 + 0.1 * seed_idx)
        
        sol = greedy_init(rng, diversity=diversity)
        
        # Quick local search
        sol = vectorized_local_search(sol, max_passes=20)
        
        cost = compute_cost(sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sol.copy()
    
    # Phase 2: Deeper local search on the best solution
    if best_solution is not None and time.time() < deadline - 15:
        improved = vectorized_local_search(best_solution, max_passes=50)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved.copy()
    
    # Phase 3: Simulated annealing on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        rng_sa = random.Random(99999)
        global rng
        rng = rng_sa
        sa_sol = simulated_annealing(best_solution, max_iter=3000)
        cost = compute_cost(sa_sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sa_sol.copy()
        
        # Another local search after SA
        if time.time() < deadline - 5:
            improved = vectorized_local_search(best_solution, max_passes=30)
            cost = compute_cost(improved)
            if cost < best_cost:
                best_cost = cost
                best_solution = improved.copy()
    
    # Fallback: if no valid solution, do a simple greedy
    if best_solution is None:
        sol = greedy_init(random.Random(0), diversity=0.0)
        best_solution = sol
        best_cost = compute_cost(sol)
    
    return best_solution.astype(int)
# EVOLVE-BLOCK-END
