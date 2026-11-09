# EVOLVE-BLOCK-START
import numpy as np
import heapq

def _solve_mcf(demands, capacities, assign_cost, open_facilities):
    """
    Solves the min-cost flow problem for a given set of open facilities.
    Returns (total_assignment_cost, assignment_array)
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if not open_facilities:
        return np.inf, None

    open_indices = list(open_facilities)
    
    # Node indices:
    # 0: Source
    # 1..n_customers: Customers
    # n_customers+1..n_customers+n_facilities: Facilities
    # n_customers+n_facilities+1: Sink
    N = n_customers + n_facilities + 2
    sink_node = N - 1
    
    # Adjacency list for residual graph
    # Each edge: [to, cap, cost, rev_index]
    graph = [[] for _ in range(N)]
    
    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
    # Source to Customers
    for i in range(n_customers):
        add_edge(0, i + 1, 1, 0)
        
    # Customers to Facilities (only open ones)
    for i in range(n_customers):
        for f in open_indices:
            add_edge(i + 1, n_customers + f + 1, 1, assign_cost[i, f])
            
    # Facilities to Sink
    for f in open_indices:
        add_edge(n_customers + f + 1, sink_node, capacities[f], 0)
        
    # Successive Shortest Path
    flow = 0
    total_cost = 0.0
    potentials = [0.0] * N
    assignment = [-1] * n_customers
    
    for _ in range(n_customers):
        # Dijkstra with potentials
        dist = [float('inf')] * N
        prev_node = [-1] * N
        prev_edge = [-1] * N
        dist[0] = 0.0
        pq = [(0.0, 0)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for k, edge in enumerate(graph[u]):
                v, cap, cost, rev = edge
                if cap > 0:
                    nd = d + cost + potentials[u] - potentials[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = k
                        heapq.heappush(pq, (nd, v))
                        
        if dist[sink_node] == float('inf'):
            return np.inf, None
            
        # Update potentials
        for i in range(N):
            if dist[i] < float('inf'):
                potentials[i] += dist[i]
                
        # Augment flow by 1
        # Calculate path cost and update residuals
        path_cost = 0.0
        v = sink_node
        while v != 0:
            u = prev_node[v]
            k = prev_edge[v]
            path_cost += graph[u][k][2]
            graph[u][k][1] -= 1
            rev_idx = graph[u][k][3]
            graph[v][rev_idx][1] += 1
            v = u
            
        total_cost += path_cost
        
        # Record assignment
        f_node = prev_node[sink_node]
        c_node = prev_node[f_node]
        i = c_node - 1
        f = f_node - n_customers - 1
        assignment[i] = f
        
    return total_cost, assignment

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Start with all facilities open
    open_set = set(range(n_facilities))
    
    # Initial solve
    current_flow_cost, current_assignment = _solve_mcf(demands, capacities, assign_cost, open_set)
    
    if current_assignment is None:
        # Fallback: should not happen, but return zeros to be safe
        return np.zeros(n_customers, dtype=int)
        
    current_total = current_flow_cost + sum(open_cost[f] for f in open_set)
    
    # Iterative improvement: try closing one facility at a time
    improved = True
    while improved:
        improved = False
        for f in list(open_set):
            new_open_set = open_set - {f}
            new_flow_cost, new_assignment = _solve_mcf(demands, capacities, assign_cost, new_open_set)
            
            if new_assignment is None:
                continue
                
            new_opening_cost = sum(open_cost[g] for g in new_open_set)
            new_total = new_flow_cost + new_opening_cost
            
            if new_total < current_total - 1e-9:
                current_total = new_total
                current_flow_cost = new_flow_cost
                current_assignment = new_assignment
                open_set = new_open_set
                improved = True
                break # Restart loop with new set
                
    return current_assignment
# EVOLVE-BLOCK-END
