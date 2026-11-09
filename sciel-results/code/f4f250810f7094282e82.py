import time
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort neighbors to make search deterministic and potentially faster
    for u in adj:
        adj[u].sort()
        
    all_vertices = set(range(n))
    
    def find_ham_cycle(adj, forbidden_edges):
        """
        Finds a Hamiltonian cycle in the graph defined by adj, 
        avoiding edges in forbidden_edges.
        Returns a list of vertices representing the cycle, or None.
        """
        start = 0
        path = [start]
        visited = {start}
        
        def backtrack():
            if len(path) == n:
                # Check if last node connects to start
                last = path[-1]
                if start in adj[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                    return True
                return False
            
            last = path[-1]
            for neighbor in adj[last]:
                if neighbor not in visited:
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

    cycles = []
    used_edges = set()
    
    # We want to find 3 cycles.
    # Strategy: 
    # 1. Find C1.
    # 2. Add C1 edges to used_edges.
    # 3. Find C2 in remaining graph.
    # 4. Add C2 edges to used_edges.
    # 5. Check if remaining graph is a single Hamiltonian cycle.
    #    If not, we might need to backtrack. 
    #    Given the small size, we can try a few variations of C1 if the greedy approach fails.
    #    However, for 6-regular graphs on 21 vertices, a solution of 3 is likely to exist 
    #    and often greedy works or a slight variation does.
    
    # Let's try to find 3 cycles.
    # We will iterate over possible starting edges for the first cycle to increase diversity if needed?
    # Actually, just finding one valid decomposition is enough.
    
    # Try to find 3 cycles
    # We'll use a simple retry mechanism with different starting points or just rely on the deterministic backtrack.
    # If the first attempt fails to produce 3, we can try shuffling the neighbor order?
    # But determinism is good. Let's just try.
    
    # To be robust, we can try to find the cycles in a way that if step 3 fails, 
    # we don't necessarily restart from scratch but we can try a different C2.
    # However, implementing full backtracking for 3 cycles is complex.
    # Let's assume the greedy "find C1, find C2, check C3" works for this specific graph.
    # If not, we return what we found (2 cycles).
    
    # Let's add a small randomization to neighbor order to handle cases where deterministic backtrack gets stuck in a bad local minimum for the 2nd/3rd cycle?
    # No, keep it simple first.
    
    c1 = find_ham_cycle(adj, used_edges)
    if c1:
        cycles.append(c1)
        for i in range(len(c1)):
            u = c1[i]
            v = c1[(i+1)%len(c1)]
            used_edges.add((min(u,v), max(u,v)))
            
    c2 = find_ham_cycle(adj, used_edges)
    if c2:
        cycles.append(c2)
        for i in range(len(c2)):
            u = c2[i]
            v = c2[(i+1)%len(c2)]
            used_edges.add((min(u,v), max(u,v)))
            
    # Check for C3
    # The remaining edges should form a 2-regular graph.
    # If it's a single cycle, we are done.
    # Let's construct the remaining graph
    rem_adj = {i: [] for i in range(n)}
    for u, v in edges:
        if (min(u,v), max(u,v)) not in used_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
    # Check if rem_adj is a single cycle of length n
    # Since it's 2-regular, it's a union of cycles.
    # We just need to check if it's connected.
    if all(len(nei) == 2 for nei in rem_adj.values()):
        # Start from 0
        visited_rem = set()
        curr = 0
        path_rem = []
        while curr not in visited_rem:
            visited_rem.add(curr)
            path_rem.append(curr)
            # Pick a neighbor not yet visited in this path (except closing)
            # Since degree is 2, one neighbor is prev, one is next.
            if len(path_rem) == 1:
                next_node = rem_adj[curr][0]
            else:
                prev_node = path_rem[-2]
                next_node = rem_adj[curr][0] if rem_adj[curr][1] == prev_node else rem_adj[curr][1]
            curr = next_node
            
        if len(path_rem) == n and rem_adj[path_rem[-1]][0] == path_rem[0] or rem_adj[path_rem[-1]][1] == path_rem[0]:
             # Ensure it closes
             if path_rem[-1] != 0: # 0 is start
                 # Check connection
                 if 0 in rem_adj[path_rem[-1]]:
                     cycles.append(path_rem)
                     # Verify disjointness just in case (should be by construction)
                     
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
