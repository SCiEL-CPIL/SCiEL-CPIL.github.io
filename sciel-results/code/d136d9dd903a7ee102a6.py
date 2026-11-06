import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limits
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        edge_set = set(tuple(e) for e in edges)
        m = len(edges)
        
        # State representation:
        # 0: Edge is a Singleton piece
        # 1: Edge is part of a Cycle piece
        # We need to ensure that edges marked '1' form valid cycles (each vertex degree 2 in the subgraph).
        
        def get_pieces(state):
            # state is a dict edge -> 0 or 1
            # Count singletons
            count = sum(1 for v in state.values() if v == 0)
            
            # Count cycles in the subgraph of edges with state 1
            # A valid decomposition requires the subgraph of 1s to be a union of cycles.
            # If it's not, the state is invalid. We return infinity for invalid states.
            deg = defaultdict(int)
            adj = defaultdict(list)
            for e, val in state.items():
                if val == 1:
                    u, v = e
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].append(v)
                    adj[v].append(u)
            
            # Check if all vertices in the subgraph have degree 2
            for u in deg:
                if deg[u] != 2:
                    return float('inf')
            
            # Count connected components in the subgraph
            visited = set()
            cycle_count = 0
            for u in deg:
                if u not in visited:
                    # BFS/DFS
                    stack = [u]
                    visited.add(u)
                    cycle_count += 1
                    while stack:
                        curr = stack.pop()
                        for nei in adj[curr]:
                            if nei not in visited:
                                visited.add(nei)
                                stack.append(nei)
            
            return count + cycle_count

        # Initialize state: all singletons
        state = {e: 0 for e in edge_set}
        current_cost = m
        
        # Helper to generate neighbors
        # Move types:
        # 1. Toggle a single edge (0->1 or 1->0). This is a valid move if we allow temporary invalid states? 
        #    No, standard SA requires valid states or a penalty. 
        #    Better: Use "2-swap" moves that maintain validity or use a penalty function.
        #    Let's use a penalty function approach where cost = pieces + 1000 * invalidity.
        #    Invalidity = sum of |deg(u)-2| for all u in subgraph.
        
        def get_cost_with_penalty(state):
            count = sum(1 for v in state.values() if v == 0)
            
            deg = defaultdict(int)
            adj = defaultdict(list)
            for e, val in state.items():
                if val == 1:
                    u, v = e
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].append(v)
                    adj[v].append(u)
            
            invalidity = 0
            for u in deg:
                invalidity += abs(deg[u] - 2)
            
            # If valid, count cycles
            if invalidity == 0:
                visited = set()
                cycle_count = 0
                for u in deg:
                    if u not in visited:
                        stack = [u]
                        visited.add(u)
                        cycle_count += 1
                        while stack:
                            curr = stack.pop()
                            for nei in adj[curr]:
                                if nei not in visited:
                                    visited.add(nei)
                                    stack.append(nei)
                return count + cycle_count
            else:
                return count + cycle_count + 1000 * invalidity # cycle_count is 0 if invalid? No, just add penalty.
                # Actually, if invalid, we can't count cycles properly. Just return high cost.
                # Let's refine: if invalid, return count + 1000 * invalidity.
                return count + 1000 * invalidity

        current_cost = get_cost_with_penalty(state)
        best_state = dict(state)
        best_cost = current_cost
        
        # SA Parameters
        T = 50.0
        cooling_rate = 0.995
        min_T = 0.1
        
        # Precompute edge list for fast access
        edge_list = list(edge_set)
        
        while T > min_T:
            if time.time() > deadline:
                break
            
            # Generate a random move
            # 1. Toggle a random edge
            # 2. Swap two random edges? (Complex to define)
            # Let's stick to toggling edges. It explores the space of "which edges are in cycles".
            
            idx = random.randint(0, len(edge_list) - 1)
            e = edge_list[idx]
            
            old_val = state[e]
            new_val = 1 - old_val
            state[e] = new_val
            
            new_cost = get_cost_with_penalty(state)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = dict(state)
            else:
                # Revert
                state[e] = old_val
            
            T *= cooling_rate
        
        # Extract pieces from best_state
        # We need to actually construct the cycle lists.
        # Edges with 0 are singletons.
        # Edges with 1 form cycles. We need to trace them.
        
        final_pieces = []
        cycle_edges = [e for e, v in best_state.items() if v == 1]
        singleton_edges = [e for e, v in best_state.items() if v == 0]
        
        for e in singleton_edges:
            final_pieces.append(list(e))
        
        # Trace cycles from cycle_edges
        adj = defaultdict(list)
        for u, v in cycle_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        visited_edges = set()
        for e in cycle_edges:
            if e in visited_edges:
                continue
            
            # Start tracing
            u, v = e
            visited_edges.add(e)
            cycle_vertices = [u]
            curr = u
            prev = None
            # We need to traverse the cycle. 
            # Since it's a valid cycle (deg 2), we can just walk.
            # Be careful not to go back to prev immediately unless it's the end.
            
            # Simple walk:
            # curr is u. next is v.
            # Let's use a standard path finding.
            path = [u]
            curr = u
            last = None
            while True:
                # Find next neighbor not equal to last
                nexts = adj[curr]
                if len(nexts) == 2:
                    if last is None:
                        nxt = nexts[0]
                    else:
                        if nexts[0] == last:
                            nxt = nexts[1]
                        else:
                            nxt = nexts[0]
                else:
                    # Should not happen if valid
                    nxt = nexts[0]
                
                # Check if edge (curr, nxt) is visited
                edge_key = tuple(sorted((curr, nxt)))
                if edge_key in visited_edges:
                    # Closed the loop?
                    if nxt == u:
                        break
                    else:
                        # Should not happen in a simple cycle
                        break
                
                visited_edges.add(edge_key)
                path.append(nxt)
                last = curr
                curr = nxt
                
                if len(path) > n: # Safety break
                    break
            
            if len(path) >= 3:
                final_pieces.append(path)
        
        results.append(final_pieces)
        
    return results

# EVOLVE-BLOCK-END
