"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a portfolio of greedy initializations
    followed by local search (Relocate and Swap) with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n_customers = len(demands)
    n_facilities = len(capacities)

    # Precompute for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    def _greedy_seed(strategy='demand_desc'):
        """
        Generate an initial assignment using different strategies.
        strategy: 'demand_desc' (largest demand first, cheapest feasible),
                  'demand_asc' (smallest demand first, cheapest feasible),
                  'random' (random order, cheapest feasible),
                  'cheapest_first' (sort by min assign cost, cheapest feasible)
        """
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        count = np.zeros(n_facilities, dtype=int)

        if strategy == 'demand_desc':
            order = np.argsort(-demands)
        elif strategy == 'demand_asc':
            order = np.argsort(demands)
        elif strategy == 'random':
            order = np.random.permutation(n_customers)
        elif strategy == 'cheapest_first':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
        else:
            order = np.argsort(-demands)

        for i in order:
            # Find cheapest feasible facility
            best_f = -1
            best_c = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    if assign_cost[i, f] < best_c:
                        best_c = assign_cost[i, f]
                        best_f = f
            if best_f == -1:
                # Fallback: assign to cheapest regardless of capacity (shouldn't happen in valid instances)
                best_f = int(np.argmin(assign_cost[i]))
            assignment[i] = best_f
            load[best_f] += demands[i]
            count[best_f] += 1

        return assignment, load, count

    def _objective(assignment, load, count):
        """Calculate total cost: open costs + assignment costs."""
        cost = 0.0
        for f in range(n_facilities):
            if count[f] > 0:
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_customers), assignment])
        return cost

    def _local_search(assignment, load, count, current_cost):
        """
        Local search with Relocate (move) and Swap moves.
        Returns improved assignment, load, count, cost.
        """
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False

            # 1. Single Moves (Relocate)
            for i in range(n_customers):
                if time.time() > deadline - 5.0:
                    break
                fi = assignment[i]
                # Try moving customer i to every other facility f
                # Sort candidate facilities by assign_cost[i, f] to find improvements faster
                candidates = np.argsort(assign_cost[i])
                for f in candidates:
                    if f == fi:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        delta_assign = assign_cost[i, f] - assign_cost[i, fi]
                        delta_open = 0.0
                        if count[fi] == 1:
                            delta_open -= open_cost[fi]
                        if count[f] == 0:
                            delta_open += open_cost[f]
                        total_delta = delta_assign + delta_open
                        if total_delta < -1e-9:
                            # Apply move
                            assignment[i] = f
                            load[fi] -= demands[i]
                            load[f] += demands[i]
                            count[fi] -= 1
                            count[f] += 1
                            current_cost += total_delta
                            improved = True
                            break  # Move to next customer

            # 2. Swaps
            for i in range(n_customers):
                if time.time() > deadline - 5.0:
                    break
                fi = assignment[i]
                for j in range(i + 1, n_customers):
                    if time.time() > deadline - 3.0:
                        break
                    fj = assignment[j]
                    if fi == fj:
                        continue

                    # Check capacity constraints for swap
                    # fi: loses i, gains j -> load[fi] - demands[i] + demands[j] <= capacities[fi]
                    # fj: loses j, gains i -> load[fj] - demands[j] + demands[i] <= capacities[fj]
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]

                    if new_load_fi <= capacities[fi] + 1e-6 and new_load_fj <= capacities[fj] + 1e-6:
                        # Calculate assignment cost delta
                        delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])

                        # Open cost delta: counts for fi and fj don't change (i leaves fi, j enters fi; j leaves fj, i enters fj)
                        # So delta_open = 0.0

                        if delta_assign < -1e-9:
                            # Apply swap
                            assignment[i] = fj
                            assignment[j] = fi
                            load[fi] = new_load_fi
                            load[fj] = new_load_fj
                            # Counts unchanged
                            current_cost += delta_assign
                            improved = True
                            break  # Break inner loop for this i, continue with next i

        return assignment, load, count, current_cost

    # Portfolio of initializations
    strategies = ['demand_desc', 'demand_asc', 'random', 'cheapest_first']
    best_cost = float('inf')
    best_assignment = None
    best_load = None
    best_count = None

    for strategy in strategies:
        if time.time() > deadline - 20.0:
            break

        assignment, load, count = _greedy_seed(strategy)
        current_cost = _objective(assignment, load, count)

        # Local search
        assignment, load, count, current_cost = _local_search(assignment, load, count, current_cost)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()
            best_load = load.copy()
            best_count = count.copy()

    # If no valid assignment found (shouldn't happen), return the first one
    if best_assignment is None:
        best_assignment, best_load, best_count = _greedy_seed('demand_desc')

    return best_assignment

# EVOLVE-BLOCK-END
