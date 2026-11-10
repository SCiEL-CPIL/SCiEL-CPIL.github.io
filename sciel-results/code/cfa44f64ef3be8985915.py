"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check if graph is regular and what degree
    # It appears to be 6-regular based on manual check of vertex 0.
    
    def find_hamiltonian_cycle(adj_in):
        """
        Finds a Hamiltonian cycle in the graph represented by adj_in.
        Returns a list of vertices or None.
        Uses backtracking with degree-based pruning.
        """
        # Deep copy to allow modification
        adj_work = {v: set(neighs) for v, neighs in adj_in.items()}
        
        # Initial check: all degrees must be >= 2
        if any(len(neighs) < 2 for neighs in adj_work.values()):
            return None
            
        # We fix the start vertex to 0 to break rotational symmetry
        start = 0
        path = [start]
        visited = {start}
        
        # Remove start from consideration for "next" steps until the end?
        # No, we just don't visit it.
        
        def backtrack(path, visited, adj_state):
            if len(path) == n:
                last = path[-1]
                if start in adj_state[last]:
                    return list(path)
                return None
                
            last = path[-1]
            
            # Pruning: Check if any unvisited vertex has degree < 2 in adj_state
            # Note: adj_state contains edges between all nodes, but we only care 
            # about connectivity for unvisited nodes.
            # Actually, if an unvisited node v has degree < 2 in the *remaining* graph 
            # (where remaining means edges not yet used in path, but path edges are 
            # only incident to 'last' and 'start' effectively?), it's complex.
            
            # Simpler pruning: 
            # For any unvisited vertex v, count its neighbors in adj_state.
            # If count < 2, return None.
            for v in range(n):
                if v not in visited:
                    if len(adj_state[v]) < 2:
                        return None
            
            # Get neighbors of last
            # Sort by degree of neighbor (Warnsdorff)
            neighbors = sorted(adj_state[last], key=lambda x: len(adj_state[x]))
            
            for nxt in neighbors:
                if nxt in visited:
                    continue
                
                # Cannot go to start unless it's the last step
                if nxt == start:
                    if len(path) == n - 1:
                        # Check if it closes
                        # This is handled in base case, but we can add it here
                        pass
                    else:
                        continue
                        
                # Create new state
                # We need to remove the edge (last, nxt) from the graph
                # to ensure edge-disjointness within this cycle? 
                # No, a simple cycle doesn't reuse edges. 
                # But we are finding *one* cycle. The edges used in this cycle 
                # are removed from the graph for the *next* cycle.
                # Within the search for ONE cycle, we just need a simple path.
                # So we just mark visited. We don't need to remove edges from 
                # adj_state for the purpose of finding the *current* cycle, 
                # UNLESS we want to prevent re-entering a node via a different edge.
                # Since it's a simple path, visited prevents re-entry.
                # However, the "degree < 2" pruning relies on the *remaining* 
                # available edges. If we don't remove used edges, the degree 
                # counts are static and the pruning is weaker.
                
                # To make pruning effective, we SHOULD remove the edge (last, nxt) 
                # from adj_state. This simulates that this edge is "consumed" 
                # and cannot be used to enter/exit other nodes? 
                # Actually, in a path, edges are used sequentially.
                # If we remove (last, nxt), then `last` loses a neighbor, and `nxt` loses a neighbor.
                # This helps detect if `nxt` or other nodes become isolated.
                
                new_adj = {v: set(neighs) for v, neighs in adj_state.items()}
                new_adj[last].discard(nxt)
                new_adj[nxt].discard(last)
                
                res = backtrack(path + [nxt], visited | {nxt}, new_adj)
                if res:
                    return res
            
            return None
        
        return backtrack(path, visited, adj_work)

    # Step 1: Find first Hamiltonian cycle
    cycle1 = find_hamiltonian_cycle(adj)
    if cycle1 is None:
        return [] # Should not happen for this graph

    # Remove edges of cycle1
    adj2 = {v: set(neighs) for v, neighs in adj.items()}
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        adj2[u].discard(v)
        adj2[v].discard(u)
        
    # Step 2: Find second Hamiltonian cycle
    cycle2 = find_hamiltonian_cycle(adj2)
    if cycle2 is None:
        return [cycle1]

    # Remove edges of cycle2
    adj3 = {v: set(neighs) for v, neighs in adj2.items()}
    for i in range(len(cycle2)):
        u = cycle2[i]
        v = cycle2[(i + 1) % len(cycle2)]
        adj3[u].discard(v)
        adj3[v].discard(u)
        
    # Step 3: Check if remaining graph is a single Hamiltonian cycle
    # A 2-regular graph is a union of cycles. 
    # We check if it's a single cycle of length n.
    
    # Check if all degrees are 2
    valid_3 = True
    for v in range(n):
        if len(adj3[v]) != 2:
            valid_3 = False
            break
            
    if valid_3:
        # Traverse to see if it's one cycle
        visited = set()
        current = 0
        cycle3 = []
        while current not in visited:
            visited.add(current)
            cycle3.append(current)
            # Pick next neighbor
            # Since degree is 2, we can just pick any unvisited neighbor, 
            # or track the previous to avoid backtracking.
            # Simpler: just follow the path.
            # Since it's a 2-regular graph, we can just walk.
            # We need to ensure we don't jump to the start prematurely.
            
            # Let's use a standard walk
            # We are at current. We came from prev (except start).
            # If we use a set to track visited, we can just pick a neighbor 
            # that is not the one we came from.
            
            pass # The logic above is a bit messy. Let's do it properly.
            
        # Proper check:
        visited_nodes = set()
        curr = 0
        prev = -1
        while curr not in visited_nodes:
            visited_nodes.add(curr)
            # Find next node
            neighbors = list(adj3[curr])
            # If curr is 0 and we just started, pick any.
            # Otherwise, pick the one that is not prev.
            if prev != -1:
                # One neighbor is prev, pick the other
                for nb in neighbors:
                    if nb != prev:
                        next_node = nb
                        break
                else:
                    # Both are prev? Impossible in simple graph unless loop.
                    valid_3 = False
                    break
            else:
                next_node = neighbors[0]
                
            if len(visited_nodes) == n and next_node != 0:
                valid_3 = False
                break
            if len(visited_nodes) < n and next_node == 0:
                valid_3 = False
                break
                
            prev = curr
            curr = next_node
            
        if len(visited_nodes) == n and curr == 0:
             # We formed a cycle. 
             # We need to extract the cycle order.
             # The `cycle3` list construction above was skipped. 
             # Let's reconstruct or just return the set? 
             # No, we need the list of vertices.
             
             # Let's re-do the traversal to get the list.
             cycle_list = []
             curr = 0
             prev = -1
             while curr not in cycle_list:
                 cycle_list.append(curr)
                 neighbors = list(adj3[curr])
                 if prev != -1:
                     next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
                 else:
                     next_node = neighbors[0]
                 prev = curr
                 curr = next_node
                 
             if curr == 0 and len(cycle_list) == n:
                 cycle3 = cycle_list
             else:
                 valid_3 = False
        else:
            valid_3 = False

    if valid_3:
        return [cycle1, cycle2, cycle3]
    else:
        return [cycle1, cycle2]

# EVOLVE-BLOCK-END
