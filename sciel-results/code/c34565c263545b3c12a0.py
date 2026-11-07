"""Branch-and-bound over facility subsets with min-cost max-flow assignment oracle for CFLP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: Branch-and-bound over the 2^20 facility-opening subsets.
    For each subset of open facilities, solve the exact capacitated assignment
    via min-cost max-flow. Use Lagrangian lower bound to prune branches.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    total_demand = demands.sum()

    def mcmf_solve(open_mask):
        """
        Solve min-cost flow: source -> customers (demand) -> facilities (capacity) -> sink.
        Returns (total_cost, assignment) or (inf, None) if infeasible.
        open_mask: boolean array of which facilities are open.
        """
        # Nodes:
        # 0 = source
        # 1..n_customers = customers
        # n_customers+1..n_customers+n_facilities = facilities
        # n_customers+n_facilities+1 = sink
        N = 1 + n_customers + n_facilities + 1
        src = 0
        sink = N - 1

        # Build edge list: (from, to, cap, cost)
        # We'll use a simple successive shortest path (Bellman-Ford / SPFA) for MCMF
        # But for speed, let's use a more efficient approach.
        
        # Actually, let's use scipy or a custom MCMF.
        # Given constraints (100 customers, 20 facilities), we can use a simpler approach:
        # This is a transportation problem with upper bounds on facility columns.
        # We can solve it as a linear program or use Hungarian-like methods.
        
        # For 100x20, let's use scipy.optimize.linprog or a custom solver.
        # Actually, let's implement a simple min-cost max-flow using SPFA.
        
        # Edge structure: list of [to, cap, cost, rev_index]
        graph = [[] for _ in range(N)]
        
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
        # Source to customers: capacity = demand[c], cost = 0
        for c in range(n_customers):
            add_edge(src, 1 + c, int(demands[c]), 0)
        
        # Customers to open facilities: capacity = demand[c], cost = assign_cost[c, f]
        open_facilities = [f for f in range(n_facilities) if open_mask[f]]
        for c in range(n_customers):
            for f in open_facilities:
                add_edge(1 + c, 1 + n_customers + f, int(demands[c]), assign_cost[c, f])
        
        # Facilities to sink: capacity = capacities[f], cost = 0
        for f in open_facilities:
            add_edge(1 + n_customers + f, sink, int(capacities[f]), 0)
        
        # Now run min-cost max-flow
        INF = float('inf')
        total_flow = 0
        total_cost = 0.0
        
        while True:
            # Find shortest path from src to sink in residual graph
            dist = [INF] * N
            in_queue = [False] * N
            prev_node = [-1] * N
            prev_edge = [-1] * N
            
            dist[src] = 0
            queue = deque([src])
            in_queue[src] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for i, (v, cap, cost, _) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev_node[v] = u
                        prev_edge[v] = i
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
            
            if dist[sink] == INF:
                break
            
            # Find bottleneck
            min_cap = INF
            v = sink
            while v != src:
                u = prev_node[v]
                i = prev_edge[v]
                min_cap = min(min_cap, graph[u][i][1])
                v = u
            
            # Augment
            total_flow += min_cap
            total_cost += min_cap * dist[sink]
            v = sink
            while v != src:
                u = prev_node[v]
                i = prev_edge[v]
                graph[u][i][1] -= min_cap
                rev = graph[u][i][3]
                graph[v][rev][1] += min_cap
                v = u
        
        if total_flow < int(total_demand) - 1:
            return INF, None
        
        # Extract assignment
        assignment = np.full(n_customers, -1, dtype=int)
        for c in range(n_customers):
            for i, (v, cap, cost, _) in enumerate(graph[1 + c]):
                if v > 1 + n_customers and cap < int(demands[c]):
                    # This edge was used
                    f = v - 1 - n_customers
                    assignment[c] = f
                    break
        
        if np.any(assignment < 0):
            return INF, None
        
        return total_cost, assignment

    def lagrangian_lb(open_mask, mu):
        """
        Lower bound: for each customer, min over open facilities of (assign_cost + mu_f).
        Plus open_cost for open facilities minus mu @ capacities for open facilities.
        """
        if not np.any(open_mask):
            return INF
        modified = np.where(open_mask[None, :], assign_cost + mu[None, :], np.inf)
        best_val = modified.min(axis=1)
        if np.any(np.isinf(best_val)):
            return INF
        fixed_charge = open_cost[open_mask].sum()
        return best_val.sum() + fixed_charge - mu[open_mask] @ capacities[open_mask]

    def greedy_initial(open_mask=None):
        """Quick greedy to get an initial UB."""
        if open_mask is None:
            # Try all facilities open
            open_mask = np.ones(n_facilities, dtype=bool)
        
        order = np.argsort(-demands)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        feasible = True
        for c in order:
            best_f = -1
            best_cost = INF
            for f in range(n_facilities):
                if not open_mask[f]:
                    continue
                if load[f] + demands[c] <= capacities[f] + 1e-9:
                    cost = assign_cost[c, f]
                    if load[f] == 0:
                        cost += open_cost[f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f < 0:
                feasible = False
                break
            result[c] = best_f
            load[best_f] += demands[c]
        
        if not feasible or np.any(result < 0):
            return None, INF
        
        # Compute actual cost
        actual_cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = result[c]
            actual_cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                actual_cost += open_cost[f]
        
        return result, actual_cost

    # Get initial upper bound
    best_solution, best_cost = greedy_initial()
    if best_solution is None:
        # Try with all facilities
        best_solution, best_cost = greedy_initial(np.ones(n_facilities, dtype=bool))
    if best_solution is None:
        # Fallback: assign to facility 0
        best_solution = np.zeros(n_customers, dtype=int)
        best_cost = INF

    # Branch and bound
    # State: (open_mask, closed_mask)
    # We'll use a priority queue ordered by lower bound
    
    import heapq
    
    # Initialize mu for Lagrangian bound
    mu = np.zeros(n_facilities)
    for f in range(n_facilities):
        if capacities[f] > 0:
            mu[f] = open_cost[f] / capacities[f] * 0.5

    # Start with the full set of facilities
    full_mask = np.ones(n_facilities, dtype=bool)
    
    # Compute initial LB for the full set
    lb_full = lagrangian_lb(full_mask, mu)
    
    # Priority queue: (lower_bound, open_mask_tuple)
    # We explore subsets by closing one facility at a time
    # Start: all open
    heap = []
    heapq.heappush(heap, (lb_full, tuple(full_mask.astype(int))))
    
    visited = set()
    visited.add(tuple(full_mask.astype(int)))
    
    # Also try the greedy solution's facility set
    if best_solution is not None:
        used_fac = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            used_fac[best_solution[c]] = True
        mask_tuple = tuple(used_fac.astype(int))
        if mask_tuple not in visited:
            lb = lagrangian_lb(used_fac, mu)
            heapq.heappush(heap, (lb, mask_tuple))
            visited.add(mask_tuple)
    
    nodes_explored = 0
    
    while heap:
        if time.time() > deadline - 10:
            break
        
        lb, mask_tuple = heapq.heappop(heap)
        if lb >= best_cost - 1e-9:
            continue
        
        mask = np.array(mask_tuple, dtype=bool)
        nodes_explored += 1
        
        # Solve MCMF for this subset
        if time.time() > deadline - 15:
            break
        
        flow_cost, flow_assign = mcmf_solve(mask)
        
        if flow_assign is not None:
            # Compute actual cost with fixed charges
            actual_cost = flow_cost
            used = np.zeros(n_facilities, dtype=bool)
            for c in range(n_customers):
                used[flow_assign[c]] = True
            for f in range(n_facilities):
                if used[f]:
                    actual_cost += open_cost[f]
            
            if actual_cost < best_cost - 1e-9:
                best_cost = actual_cost
                best_solution = flow_assign.copy()
        
        # Generate children: close one more facility
        open_indices = [f for f in range(n_facilities) if mask[f]]
        for f in open_indices:
            if time.time() > deadline - 10:
                break
            new_mask = mask.copy()
            new_mask[f] = False
            new_tuple = tuple(new_mask.astype(int))
            
            if new_tuple in visited:
                continue
            visited.add(new_tuple)
            
            # Check if remaining capacity can serve demand
            remaining_cap = capacities[new_mask].sum()
            if remaining_cap < total_demand - 1e-9:
                continue
            
            # Compute LB
            child_lb = lagrangian_lb(new_mask, mu)
            if child_lb < best_cost - 1e-9:
                heapq.heappush(heap, (child_lb, new_tuple))
    
    # Final local search on best solution
    if best_solution is not None and time.time() < deadline - 5:
        # Simple swap local search
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(best_solution[c])] += demands[c]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for c in range(n_customers):
                old_f = int(best_solution[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    best_solution[c] = best_new_f
                    load[best_new_f] += demands[c]
                    improved = True
                else:
                    load[old_f] += demands[c]
    
    if best_solution is None:
        best_solution = np.zeros(n_customers, dtype=int)
    
    return best_solution.astype(int)
# EVOLVE-BLOCK-END
