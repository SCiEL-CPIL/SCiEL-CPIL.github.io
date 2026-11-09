"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute adjacency: adj[i] is a set of nodes that CONFLICT with i
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = i + 1, j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Tomita's algorithm for Maximum Independent Set
    # We want to find a maximum set of nodes with no conflicts (independent set in conflict graph)
    
    # Use bitmasks for efficiency
    # adj_mask[i] is a bitmask of nodes conflicting with i
    adj_mask = [0] * N
    for i in range(N):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Initial greedy solution: all odds
    odds_mask = 0
    for i in range(N):
        if (i + 1) % 2 == 1:
            odds_mask |= (1 << i)
    
    best_mask = odds_mask
    best_size = bin(odds_mask).count('1')
    
    # Tomita's branch and bound with coloring
    def color_sort(candidates_mask):
        """
        Compute a greedy coloring of the subgraph induced by candidates_mask.
        Returns (coloring, order) where coloring[i] is the color class index for node i,
        and order is a list of nodes sorted by color class (highest color first).
        """
        # Get list of candidate nodes
        nodes = []
        m = candidates_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            m ^= lsb
        
        # Greedy coloring: assign smallest possible color
        color_of = {}
        color_classes = []  # list of lists
        used_colors = [0] * len(nodes)  # color used by each node
        
        for node in nodes:
            # Find smallest color not used by neighbors in candidates
            neighbor_colors = set()
            for nb in nodes:
                if nb != node and (adj_mask[node] >> nb) & 1:
                    if nb in color_of:
                        neighbor_colors.add(color_of[nb])
            
            c = 0
            while c in neighbor_colors:
                c += 1
            
            color_of[node] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(node)
        
        # Order nodes by color class (highest first)
        order = []
        for c in range(len(color_classes) - 1, -1, -1):
            order.extend(color_classes[c])
        
        return color_of, order, len(color_classes)
    
    def tomita(candidates_mask, current_mask, current_size):
        """
        Tomita's branch and bound.
        candidates_mask: bitmask of remaining candidate nodes
        current_mask: bitmask of nodes already in the independent set
        current_size: size of current independent set
        Returns (best_mask, best_size)
        """
        nonlocal best_mask, best_size
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Upper bound: current_size + number of candidate nodes
        num_candidates = bin(candidates_mask).count('1')
        if current_size + num_candidates <= best_size:
            return
        
        # Get color bound
        color_of, order, num_colors = color_sort(candidates_mask)
        
        # Upper bound from coloring: current_size + num_colors
        if current_size + num_colors <= best_size:
            return
        
        # Branch on nodes in reverse color order
        for node in reversed(order):
            if time.time() > deadline:
                return
            
            # Prune: if current_size + color_of[node] <= best_size, skip
            if current_size + color_of[node] <= best_size:
                return
            
            # Branch: include node
            new_current = current_mask | (1 << node)
            # Candidates that can still be added: those not conflicting with node
            new_candidates = candidates_mask & ~(adj_mask[node] | (1 << node))
            
            tomita(new_candidates, new_current, current_size + 1)
            
            # Remove node from candidates
            candidates_mask &= ~(1 << node)
            
            # Update color bound after removal
            # Recompute if needed (simplified: just continue)
    
    # Run Tomita's algorithm
    tomita((1 << N) - 1, 0, 0)
    
    # If we didn't find anything better, use the greedy solution
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    
    # Verify the solution
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Invalid, fall back to odds
                result = [x + 1 for x in range(N) if (x + 1) % 2 == 1]
                break
    
    return result
# EVOLVE-BLOCK-END
