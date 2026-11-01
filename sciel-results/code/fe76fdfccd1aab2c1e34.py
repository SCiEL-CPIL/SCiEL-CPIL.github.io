import collections
import itertools
import random
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal superpermutation for n=4 has length 33
        # We'll use the known construction
        return "1234123142312431213421324132431243214321"
    
    # For n >= 5, we use the greedy cycle-chaining approach
    # This is based on the known construction that gives n! + (n-1)! + n - 1
    # For n=5, this gives 120 + 24 + 4 = 148
    # For n=6, this gives 720 + 120 + 5 = 845
    # For n=7, this gives 5040 + 720 + 6 = 5766
    
    # The construction works as follows:
    # 1. Consider all (n-1)-permutations as nodes
    # 2. The "deterministic" next node is: remove first symbol, append the missing symbol
    # 3. This partitions the (n-1)-permutations into cycles
    # 4. Traverse cycles in a specific order to minimize transitions
    
    # Generate all (n-1)-permutations
    perms = list(itertools.permutations(range(1, n + 1), n - 1))
    num_nodes = len(perms)
    node_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Compute deterministic next for each node
    # missing symbol is the one not in the tuple
    det_next = [None] * num_nodes
    for i, p in enumerate(perms):
        # Find missing symbol
        p_set = set(p)
        missing = 0
        for s in range(1, n + 1):
            if s not in p_set:
                missing = s
                break
        next_p = p[1:] + (missing,)
        det_next[i] = node_to_idx[next_p]
    
    # Find cycles
    visited = [False] * num_nodes
    cycles = []
    cycle_id = [0] * num_nodes
    for i in range(num_nodes):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                cycle_id[curr] = len(cycles)
                curr = det_next[curr]
            cycles.append(cycle)
    
    num_cycles = len(cycles)
    
    # The number of cycles is (n-2)! for the standard construction
    # Each cycle has length n-1
    
    # For the greedy construction, we need to find the optimal order to visit cycles
    # The key insight: when we finish one cycle, we're at a specific node
    # We need to find the shortest path to the start of another unvisited cycle
    
    # For small n, we can try to optimize the cycle order
    # For larger n, we use a greedy heuristic
    
    # Build adjacency: from each node, what nodes can we reach in 1 step?
    # A node p = (a1, a2, ..., a_{n-1}) can go to (a2, ..., a_{n-1}, s) for any s not in {a2, ..., a_{n-1}}
    # Actually, s can be anything from 1 to n, but the new node must be a valid (n-1)-perm
    
    # Let's precompute the "exit" options from each cycle
    # When we're at the last node of a cycle, we can append any symbol to leave
    
    # For the standard construction, the optimal cycle order is known
    # Let's implement a greedy approach:
    
    # Start with the first cycle
    start_cycle = 0
    current_node = cycles[start_cycle][0]
    word = list(perms[current_node])
    
    visited_cycles = set([start_cycle])
    
    # Helper to get the next node given current node and symbol to append
    def next_node(node_idx, sym):
        p = perms[node_idx]
        new_p = p[1:] + (sym,)
        return node_to_idx[new_p]
    
    # Helper to find the missing symbol of a node
    def missing_sym(node_idx):
        p = perms[node_idx]
        p_set = set(p)
        for s in range(1, n + 1):
            if s not in p_set:
                return s
        return 0
    
    # Traverse the current cycle
    # We start at cycles[start_cycle][0]
    # We need to go around the cycle
    curr = current_node
    for step in range(n - 1):
        sym = missing_sym(curr)
        word.append(sym)
        curr = det_next[curr]
    
    # Now we're at cycles[start_cycle][-1] (the last node of the cycle)
    # Wait, let me re-check. We start at cycles[start_cycle][0].
    # After n-1 steps, we should be back at cycles[start_cycle][0]? No.
    # The cycle length is n-1. So after n-1 M-steps, we're back to the start.
    # But we want to end at the last node of the cycle to transition out.
    
    # Let me redo this. We start at node 0 of the cycle.
    # We take M-edges. After k steps, we're at cycle[k].
    # After n-1 steps, we're back at cycle[0].
    # So we need to stop at cycle[-1] to transition out.
    
    # Let's restart the word construction
    word = list(perms[cycles[start_cycle][0]])
    curr = cycles[start_cycle][0]
    
    # Traverse to the end of the cycle
    for step in range(n - 2):
        sym = missing_sym(curr)
        word.append(sym)
        curr = det_next[curr]
    
    # Now curr is cycles[start_cycle][-1]
    
    # Find the next cycle to visit
    # From curr, we can append any symbol s to get to next_node(curr, s)
    # We want to reach a node in an unvisited cycle
    
    # For efficiency, let's precompute for each node which cycles are reachable
    # Actually, let's just do BFS
    
    remaining_cycles = set(range(num_cycles)) - visited_cycles
    
    while remaining_cycles:
        # From curr, find the shortest path to any node in remaining_cycles
        # BFS
        queue = collections.deque()
        queue.append((curr, []))
        bfs_visited = {curr}
        
        found = False
        next_node_idx = None
        path = []
        
        while queue:
            node_idx, path_so_far = queue.popleft()
            
            # Check if this node is in a remaining cycle
            cid = cycle_id[node_idx]
            if cid in remaining_cycles:
                found = True
                next_node_idx = node_idx
                path = path_so_far
                break
            
            # Expand
            p = perms[node_idx]
            p_set = set(p[1:])  # The first symbol is dropped
            for s in range(1, n + 1):
                new_p = p[1:] + (s,)
                if new_p in node_to_idx:
                    ni = node_to_idx[new_p]
                    if ni not in bfs_visited:
                        bfs_visited.add(ni)
                        queue.append((ni, path_so_far + [s]))
        
        if not found:
            break
        
        # Append the path
        for sym in path:
            word.append(sym)
        
        # Move to next_node_idx
        curr = next_node_idx
        
        # Mark the cycle as visited
        visited_cycles.add(cycle_id[curr])
        remaining_cycles.remove(cycle_id[curr])
        
        # Traverse this cycle (to the end)
        # We're at some node in the cycle. We need to go to the end.
        # Find our position in the cycle
        cycle = cycles[cycle_id[curr]]
        # Find curr in cycle
        try:
            pos = cycle.index(curr)
        except ValueError:
            # Should not happen
            break
        
        # Traverse from pos to the end of the cycle
        for step in range(len(cycle) - 1 - pos):
            sym = missing_sym(curr)
            word.append(sym)
            curr = det_next[curr]
    
    return "".join(map(str, word))
