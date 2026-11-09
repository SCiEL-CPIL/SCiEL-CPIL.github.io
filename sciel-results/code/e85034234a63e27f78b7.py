"""Capacitated facility location: min-cost flow with potentials + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n_cust = len(demands)
    n_fac = len(capacities)

    # Total demand
    total_demand = int(np.sum(demands))

    # Check feasibility: sum of capacities must cover total demand
    if np.sum(capacities) < total_demand:
        # Infeasible; return best-effort (shouldn't happen per problem statement)
        return _greedy_fallback(demands, capacities, assign_cost)

    # We model this as a min-cost flow problem.
    # Source -> Facility (capacity = capacities[f], cost = open_cost[f] if first unit, 0 otherwise)
    # Facility -> Customer (capacity = 1 per unit demand... but that's too fine-grained)
    #
    # Better formulation:
    # Since each customer must be assigned to exactly one facility, and demands are integers,
    # we can think of it as: each customer i sends demands[i] units of flow to one facility.
    # But min-cost flow is linear, so we need to handle the "open" cost.
    #
    # Alternative: Use a successive shortest path on a bipartite graph where:
    # - Source connects to each customer with capacity = demands[i], cost = 0
    # - Each customer connects to each facility with capacity = demands[i], cost = assign_cost[i,f]
    # - Each facility connects to sink with capacity = capacities[f], cost = 0
    # - Plus we need to add open_cost for facilities that get any flow.
    #
    # The open cost makes this non-linear. A common trick: add a dummy edge from source to facility
    # with capacity 1 and cost = open_cost[f], and the facility->sink edge has capacity = capacities[f]
    # but the first unit must go through the open edge. This is tricky in standard min-cost flow.
    #
    # Simpler approach: Since n_fac is only 20, we can enumerate subsets of open facilities.
    # But 2^20 = 1M which is too many.
    #
    # Better: Use min-cost flow where we add open_cost as a fixed charge. We can handle this by:
    # 1. First solve the assignment ignoring open costs (min-cost flow with just assign costs)
    # 2. Then try to close facilities that have low utilization to save open costs
    # 3. Reassign customers from closed facilities to other open ones
    #
    # Actually, let's use a proper min-cost flow with the fixed charge handled by:
    # - For each facility f, add an edge from source to f with cap=1, cost=open_cost[f]
    # - Also add an edge from source to f with cap=capacities[f], cost=0
    # - But we need to ensure that if any flow goes to f, the open edge is used.
    # This doesn't work directly in standard MCF.
    #
    # Practical approach for this small problem (100 customers, 20 facilities):
    # 1. Solve min-cost flow ignoring open costs to get initial assignment
    # 2. Identify facilities with very low load; try closing them
    # 3. Reassign their customers via min-cost flow on remaining facilities
    # 4. Repeat until no improvement
    #
    # For the min-cost flow itself, since n_cust=100, n_fac=20, we can use
    # successive shortest path with potentials.
    
    # Min-cost flow: assign customers to facilities minimizing sum of assign_cost
    # subject to capacity constraints.
    # This is a transportation problem. We can solve it efficiently.
    
    # Use scipy if available, otherwise implement MCF
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment works for square or rectangular but not for capacities > 1
        # We need a different approach
    except ImportError:
        pass
    
    # Implement min-cost flow for the transportation problem
    # Source -> Customer_i (cap=demands[i], cost=0)
    # Customer_i -> Facility_f (cap=demands[i], cost=assign_cost[i,f])
    # Facility_f -> Sink (cap=capacities[f], cost=0)
    
    # Since the graph is bipartite and small, use successive shortest path with potentials.
    # Number of nodes: 1 (source) + 100 (customers) + 20 (facilities) + 1 (sink) = 122
    # Edges: 100 (source->cust) + 100*20 (cust->fac) + 20 (fac->sink) = 2120
    
    def solve_mcf(demands, capacities, costs):
        """Solve min-cost flow: send total_demand units from source to sink.
        Returns assignment array and total cost."""
        n_c = len(demands)
        n_f = len(capacities)
        
        # Node indices: 0=source, 1..n_c=customers, n_c+1..n_c+n_f=facilities, n_c+n_f+1=sink
        src = 0
        cust_start = 1
        fac_start = 1 + n_c
        sink = fac_start + n_f
        total_nodes = sink + 1
        
        # Build adjacency list: adj[u] = list of (v, cap, cost, edge_index)
        # We need reverse edges too for residual graph
        # Use a structure: for each directed edge, store forward and backward
        # edges[u] = list of [v, cap, cost, rev_index]
        
        edges = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap, cost):
            """Add edge u->v with given capacity and cost. Returns edge index in edges[u]."""
            fwd = [v, cap, cost, len(edges[v])]
            bwd = [u, 0, -cost, len(edges[u])]
            edges[u].append(fwd)
            edges[v].append(bwd)
            return len(edges[u]) - 1
        
        # Source to customers
        for i in range(n_c):
            add_edge(src, cust_start + i, demands[i], 0)
        
        # Customers to facilities
        for i in range(n_c):
            for f in range(n_f):
                add_edge(cust_start + i, fac_start + f, demands[i], costs[i, f])
        
        # Facilities to sink
        for f in range(n_f):
            add_edge(fac_start + f, sink, capacities[f], 0)
        
        total_demand = int(np.sum(demands))
        
        # Successive shortest path with potentials (Johnson's algorithm)
        # Initial potentials: 0 (all costs are non-negative, so 0 is valid)
        potential = np.zeros(total_nodes)
        flow = 0
        total_cost = 0.0
        
        # For efficiency, we'll use Dijkstra with potentials
        # Since all reduced costs are non-negative when potentials are valid
        
        import heapq
        
        while flow < total_demand:
            # Find shortest path from src to sink in residual graph
            dist = np.full(total_nodes, np.inf)
            dist[src] = 0
            prev_node = [-1] * total_nodes
            prev_edge = [-1] * total_nodes
            visited = np.zeros(total_nodes, dtype=bool)
            
            pq = [(0, src)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                if d > dist[u]:
                    continue
                    
                for idx, (v, cap, cost, rev_idx) in enumerate(edges[u]):
                    if cap <= 0:
                        continue
                    # Reduced cost
                    rc = cost + potential[u] - potential[v]
                    new_dist = d + rc
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_node[v] = u
                        prev_edge[v] = idx
                        heapq.heappush(pq, (new_dist, v))
            
            if dist[sink] == np.inf:
                break  # No more augmenting path
            
            # Update potentials
            for i in range(total_nodes):
                if dist[i] < np.inf:
                    potential[i] += dist[i]
            
            # Find bottleneck along the path
            bottleneck = total_demand - flow
            v = sink
            while v != src:
                u = prev_node[v]
                idx = prev_edge[v]
                bottleneck = min(bottleneck, edges[u][idx][1])
                v = u
            
            # Augment flow
            v = sink
            while v != src:
                u = prev_node[v]
                idx = prev_edge[v]
                edges[u][idx][1] -= bottleneck
                # Reverse edge
                rev = edges[u][idx][3]
                edges[v][rev][1] += bottleneck
                v = u
            
            flow += bottleneck
            total_cost += bottleneck * (potential[sink] - potential[src])
        
        # Extract assignment
        assignment = np.zeros(n_c, dtype=int)
        for i in range(n_c):
            cust_node = cust_start + i
            for idx, (v, cap, cost, rev_idx) in enumerate(edges[cust_node]):
                if v >= fac_start and v < fac_start + n_f:
                    # Check if there's flow on this edge
                    # Flow = original_cap - current_cap
                    orig_cap = demands[i]
                    if orig_cap - cap > 0:
                        f = v - fac_start
                        assignment[i] = f
                        break
        
        return assignment, total_cost
    
    # Step 1: Solve MCF ignoring open costs
    assignment, mcf_cost = solve_mcf(demands, capacities, assign_cost)
    
    # Verify assignment
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    # Compute total cost including open costs
    def compute_total_cost(assignment, demands, capacities, open_cost, assign_cost):
        load = np.zeros(len(capacities))
        for i in range(len(demands)):
            load[assignment[i]] += demands[i]
        # Check capacity
        if np.any(load > capacities):
            return np.inf
        open_mask = load > 0
        total = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(len(demands)), assignment])
        return total
    
    current_cost = compute_total_cost(assignment, demands, capacities, open_cost, assign_cost)
    
    # Step 2: Try to close facilities to save open costs
    # For each facility with low load, try closing it and reassigning
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 10:
            break
        
        # Get current loads
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        
        # Try closing each open facility (sorted by load ascending, to close least utilized first)
        open_facilities = np.where(load > 0)[0]
        open_facilities = open_facilities[np.argsort(load[open_facilities])]
        
        for f_to_close in open_facilities:
            if time.time() > deadline - 10:
                break
            
            # Get customers assigned to f_to_close
            custs_to_reassign = np.where(assignment == f_to_close)[0]
            if len(custs_to_reassign) == 0:
                continue
            
            # Create modified problem: remove facility f_to_close
            # Reassign these customers to other facilities
            # Use MCF for just these customers
            n_reassign = len(custs_to_reassign)
            if n_reassign == 0:
                continue
            
            # Available capacities of other facilities
            other_facilities = [f for f in range(n_fac) if f != f_to_close]
            if len(other_facilities) == 0:
                continue
            
            # Current loads of other facilities
            other_loads = np.array([load[f] for f in other_facilities])
            other_caps = np.array([capacities[f] for f in other_facilities])
            available_cap = other_caps - other_loads
            
            if np.sum(available_cap) < np.sum(demands[custs_to_reassign]):
                continue  # Can't reassign
            
            # Cost matrix for reassigning customers
            reassign_costs = assign_cost[custs_to_reassign][:, other_facilities]
            reassign_demands = demands[custs_to_reassign]
            reassign_caps = available_cap
            
            # Solve MCF for reassignment
            new_assignment, new_cost = solve_mcf(reassign_demands, reassign_caps, reassign_costs)
            
            # Check if this improves total cost
            # Old cost contribution: open_cost[f_to_close] + sum of assign costs for these customers
            old_contrib = open_cost[f_to_close] + np.sum(assign_cost[custs_to_reassign, f_to_close])
            # New cost contribution: sum of new assign costs (no open cost for f_to_close)
            # But we might open new facilities that weren't open before
            new_load = other_loads.copy()
            for j, f_idx in enumerate(new_assignment):
                new_load[f_idx] += reassign_demands[j]
            
            # Facilities that become newly open
            newly_open = (new_load > 0) & (other_loads == 0)
            new_open_cost = np.sum(open_cost[[f for f, no in zip(other_facilities, newly_open) if no]])
            
            new_assign_cost = np.sum(reassign_costs[np.arange(n_reassign), new_assignment])
            new_contrib = new_open_cost + new_assign_cost
            
            # Also need to account for facilities that were open but might lose all customers
            # (they stay open if they still have customers)
            
            if new_contrib < old_contrib:
                # Apply the reassignment
                for j, orig_idx in enumerate(custs_to_reassign):
                    assignment[orig_idx] = other_facilities[new_assignment[j]]
                current_cost = compute_total_cost(assignment, demands, capacities, open_cost, assign_cost)
                improved = True
                break  # Restart the loop
    
    # Step 3: Local search - try moving individual customers to different facilities
    if time.time() < deadline - 5:
        # Compute current loads
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() > deadline - 5:
                break
            
            current_f = assignment[i]
            current_cost_i = assign_cost[i, current_f]
            
            # Try all other facilities
            best_f = current_f
            best_delta = 0  # We want to minimize delta = new_cost - old_cost
            
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Cost change: assign_cost[i,f] - assign_cost[i,current_f]
                delta_assign = assign_cost[i, f] - current_cost_i
                
                # Open cost change
                # If f is currently closed (load[f] == 0), we add open_cost[f]
                # If current_f becomes empty (load[current_f] - demands[i] == 0), we save open_cost[current_f]
                delta_open = 0
                if load[f] == 0:
                    delta_open += open_cost[f]
                if load[current_f] - demands[i] == 0:
                    delta_open -= open_cost[current_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f = f
            
            if best_f != current_f:
                assignment[i] = best_f
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
    
    # Final verification
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    if np.any(load > capacities):
        # Fallback: use greedy
        return _greedy_fallback(demands, capacities, assign_cost)
    
    return assignment

def _greedy_fallback(demands, capacities, assign_cost):
    """Fallback greedy assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result

# EVOLVE-BLOCK-END
