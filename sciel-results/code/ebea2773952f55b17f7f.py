import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute graph properties
n = P['n']
adj = {i: [] for i in range(n)}
all_edges = set()
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)
    # Normalize edge representation for easy set operations
    if u < v:
        all_edges.add((u, v))
    else:
        all_edges.add((v, u))

def get_edge(u, v):
    if u < v:
        return (u, v)
    else:
        return (v, u)

def find_hamiltonian_cycle(available_edges, order, start=0, time_limit=5.0, start_time=None):
    """
    Finds a Hamiltonian cycle using backtracking with a specific neighbor ordering.
    available_edges: set of tuples (u, v) with u < v.
    order: list of vertices defining the priority order for visiting neighbors.
    """
    if start_time is None:
        start_time = time.time()
        
    # Create a local adjacency list filtered by available edges
    # and sorted by the provided order
    local_adj = {i: [] for i in range(n)}
    order_map = {v: i for i, v in enumerate(order)}
    
    for u, v in available_edges:
        local_adj[u].append(v)
        local_adj[v].append(u)
        
    for i in range(n):
        local_adj[i].sort(key=lambda x: order_map.get(x, 100))

    path = [start]
    visited = [False] * n
    visited[start] = True
    current_time = start_time

    def dfs(current, depth):
        nonlocal current_time
        # Time check
        if time.time() - start_time > time_limit:
            return None
            
        if depth == n:
            # Check if closing edge is available
            edge = get_edge(current, start)
            if edge in available_edges:
                return list(path)
            return None
        
        for neighbor in local_adj[current]:
            if not visited[neighbor]:
                edge = get_edge(current, neighbor)
                if edge in available_edges:
                    visited[neighbor] = True
                    path.append(neighbor)
                    
                    result = dfs(neighbor, depth + 1)
                    if result is not None:
                        return result
                    
                    path.pop()
                    visited[neighbor] = False
                    
        return None

    return dfs(start, 1)

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s kill
    
    # Generate heuristic orders
    base_order = list(range(n))
    reverse_order = base_order[::-1]
    
    # Add a few random shuffles with fixed seeds for reproducibility
    orders = [base_order, reverse_order]
    random.seed(42)
    for _ in range(10):
        shuffled = base_order.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    best_cycles = []
    
    for order in orders:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1.0:
            break
            
        # Try to find 3 cycles with this order
        current_edges = set(all_edges)
        cycles_found = []
        
        for i in range(3):
            sub_remaining = remaining_time - (time.time() - start_time)
            if sub_remaining < 0.5:
                break
                
            # Allocate time for this cycle search
            # We want to be fast. If we find it quickly, great.
            # If not, we might need to try a different order.
            cycle = find_hamiltonian_cycle(current_edges, order, start=0, time_limit=sub_remaining - 0.1, start_time=time.time())
            
            if cycle is None:
                break
                
            # Add to found cycles
            cycles_found.append(cycle)
            
            # Remove edges from current_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_edges.discard(get_edge(u, v))
                
        if len(cycles_found) > len(best_cycles):
            best_cycles = cycles_found
            
        if len(best_cycles) == 3:
            break
            
    # If we didn't find 3, try a few more specific strategies or just return best
    # The loop above tries to maximize the count.
    
    # Final validation and type conversion
    # The task requires lists of ints.
    # find_hamiltonian_cycle returns lists of ints.
    
    if not best_cycles:
        # Extreme fallback: try one more time with no time limit constraint (just raw speed)
        # But this is unlikely to be needed if the graph is solvable.
        # If we still have nothing, we fail. But the graph is dense and 3-disjoint exists.
        pass
        
    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
