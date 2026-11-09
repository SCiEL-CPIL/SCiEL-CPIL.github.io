import time
import itertools
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # 85s limit with safety margin
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort neighbors to make search deterministic and potentially faster
    for u in adj:
        adj[u].sort()
        
    all_vertices = set(range(n))
    
    def find_ham_cycle(adj, forbidden_edges, start=0, neighbor_order=None, time_budget=1.0):
        """
        Finds a Hamiltonian cycle in the graph defined by adj, 
        avoiding edges in forbidden_edges.
        Returns a list of vertices representing the cycle, or None.
        """
        local_start_time = time.time()
        
        # Use custom neighbor order if provided, else use adj
        def get_neighbors(u):
            if neighbor_order is not None:
                return neighbor_order[u]
            return adj[u]
        
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() - local_start_time > time_budget or time.time() > deadline:
                return False
                
            if len(path) == n:
                # Check if last node connects to start
                last = path[-1]
                if start in adj[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                    return True
                return False
            
            last = path[-1]
            neighbors = get_neighbors(last)
            
            # Heuristic: sort neighbors by degree in remaining graph (Warnsdorff-like)
            if len(path) > 1 and len(path) < n - 2:
                def degree_score(nb):
                    if nb in visited:
                        return 0
                    edge_key = (min(last, nb), max(last, nb))
                    if edge_key in forbidden_edges:
                        return -1
                    # Count unvisited neighbors of nb (excluding last)
                    count = 0
                    for nn in adj[nb]:
                        if nn not in visited and nn != last:
                            ek = (min(nb, nn), max(nb, nn))
                            if ek not in forbidden_edges:
                                count += 1
                    return count
                unvisited_neighbors = [nb for nb in neighbors if nb not in visited]
                neighbors = sorted(unvisited_neighbors, key=degree_score)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                # Check if edge is forbidden
                edge_key = (min(last, neighbor), max(last, neighbor))
                if edge_key in forbidden_edges:
                    continue
                # Heuristic: if only one node left, it must connect to start
                if len(path) == n - 1:
                    if start not in adj[neighbor]:
                        continue
                    if (min(neighbor, start), max(neighbor, start)) in forbidden_edges:
                        continue
                    
                path.append(neighbor)
                visited.add(neighbor)
                if backtrack():
                    return True
                path.pop()
                visited.remove(neighbor)
            return False
            
        if backtrack():
            return path
        return None

    def try_find_multiple_cycles(num_cycles_target=3):
        """Try to find multiple edge-disjoint Hamiltonian cycles."""
        cycles = []
        used_edges = set()
        
        # Generate diverse neighbor orderings
        base_order = {u: adj[u][:] for u in range(n)}
        orderings = [base_order]
        
        # Reverse order
        rev_order = {u: adj[u][::-1] for u in range(n)}
        orderings.append(rev_order)
        
        # Randomized orders with different seeds
        rng = random.Random(123)
        for seed in range(8):
            rand_order = {}
            for u in range(n):
                lst = adj[u][:]
                rng.shuffle(lst)
                rand_order[u] = lst
            orderings.append(rand_order)
        
        # Try different start vertices
        start_vertices = list(range(n))
        rng.shuffle(start_vertices)
        
        for order_idx, order in enumerate(orderings):
            if time.time() > deadline:
                break
            if len(cycles) >= num_cycles_target:
                break
                
            for start in start_vertices:
                if time.time() > deadline:
                    break
                if len(cycles) >= num_cycles_target:
                    break
                    
                # Try to find a cycle with this start and order
                c = find_ham_cycle(adj, used_edges, start=start, neighbor_order=order, time_budget=2.0)
                if c is not None:
                    cycles.append(c)
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i+1)%len(c)]
                        used_edges.add((min(u,v), max(u,v)))
                    break  # Try next order after finding one cycle
        
        return cycles

    # First attempt: try to find 3 cycles
    cycles = try_find_multiple_cycles(num_cycles_target=3)
    
    if not cycles:
        # Fallback: just find what we can greedily
        used_edges = set()
        base_order = {u: adj[u][:] for u in range(n)}
        rev_order = {u: adj[u][::-1] for u in range(n)}
        orderings = [base_order, rev_order]
        
        rng = random.Random(42)
        for _ in range(5):
            rand_order = {}
            for u in range(n):
                lst = adj[u][:]
                rng.shuffle(lst)
                rand_order[u] = lst
            orderings.append(rand_order)
        
        for order in orderings:
            c = find_ham_cycle(adj, used_edges, start=0, neighbor_order=order, time_budget=3.0)
            if c:
                cycles.append(c)
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i+1)%len(c)]
                    used_edges.add((min(u,v), max(u,v)))
                if time.time() > deadline:
                    break
    
    # Convert to lists of ints
    result = [list(map(int, c)) for c in cycles]
    
    # Check validity: all cycles must have same vertex set
    if result:
        vset = set(result[0])
        for c in result:
            if set(c) != vset:
                return []
        # Check edge disjointness
        seen_edges = set()
        for c in result:
            for i in range(len(c)):
                u, v = c[i], c[(i+1)%len(c)]
                key = (min(u,v), max(u,v))
                if key in seen_edges:
                    return []
                seen_edges.add(key)
                
    return result

# EVOLVE-BLOCK-END
