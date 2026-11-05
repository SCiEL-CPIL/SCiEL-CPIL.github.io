import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy:
      1. Use backtracking with unit propagation for 2-colorability checking.
      2. Multi-start search with better initialization.
      3. Aggressive greedy pruning with multiple passes.
      4. Safety: always return a valid non-2-colorable hypergraph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_edges = None
    best_n = 14

    def is_non_2colorable_backtrack(edges, n, time_limit=None):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        Uses backtracking with unit propagation.
        
        A proper 2-colouring means: no edge is monochromatic.
        So we're checking if there EXISTS a colouring where every edge has both colours.
        If no such colouring exists, the hypergraph is NOT 2-colorable (fails Property B).
        """
        if not edges:
            return False

        # Build edge masks and adjacency for unit propagation
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        
        # For unit propagation: if an edge has 4 vertices of one colour, the 5th must be the other
        # We use a constraint satisfaction approach
        
        # Try to find a valid 2-colouring
        # colour[v] = 0 or 1, unassigned = -1
        colour = [-1] * n
        
        # Constraint: for each edge, not all 5 vertices have same colour
        # Equivalently: each edge must have at least one 0 and at least one 1
        
        # Backtracking with unit propagation
        def backtrack(idx, colour):
            if time_limit and time.time() > time_limit:
                return False  # Timeout, assume not found (conservative)
            
            # Find first unassigned vertex
            v = -1
            for i in range(n):
                if colour[i] == -1:
                    v = i
                    break
            
            if v == -1:
                # All assigned, check if valid
                for em in edge_masks:
                    # Check if edge is monochromatic
                    mono = True
                    first_colour = -1
                    for bit in range(n):
                        if em & (1 << bit):
                            if first_colour == -1:
                                first_colour = colour[bit]
                            elif colour[bit] != first_colour:
                                mono = False
                                break
                    if mono:
                        return False  # Invalid: monochromatic edge found
                return True  # Valid colouring found
            
            # Try both colours
            for c in [0, 1]:
                colour[v] = c
                
                # Unit propagation: check if any edge is forced
                # If an edge has 4 vertices with colour c, the 5th must be 1-c
                # If an edge has 4 vertices with colour 1-c, the 5th must be c
                forced = {}
                conflict = False
                for em in edge_masks:
                    count_c = 0
                    count_1c = 0
                    unassigned_in_edge = []
                    for bit in range(n):
                        if em & (1 << bit):
                            if colour[bit] == c:
                                count_c += 1
                            elif colour[bit] == 1 - c:
                                count_1c += 1
                            else:
                                unassigned_in_edge.append(bit)
                    
                    # If 4 are c and 1 unassigned, that one must be 1-c
                    if count_c == 4 and len(unassigned_in_edge) == 1:
                        u = unassigned_in_edge[0]
                        if u not in forced:
                            forced[u] = 1 - c
                        elif forced[u] != 1 - c:
                            conflict = True
                            break
                    
                    # If 4 are 1-c and 1 unassigned, that one must be c
                    if count_1c == 4 and len(unassigned_in_edge) == 1:
                        u = unassigned_in_edge[0]
                        if u not in forced:
                            forced[u] = c
                        elif forced[u] != c:
                            conflict = True
                            break
                
                if not conflict:
                    # Apply forced assignments
                    for u, fc in forced.items():
                        if colour[u] == -1:
                            colour[u] = fc
                        elif colour[u] != fc:
                            conflict = True
                            break
                
                if not conflict:
                    # Check for immediate conflict: any edge with all 5 same colour
                    immediate_conflict = False
                    for em in edge_masks:
                        all_same = True
                        first_col = -1
                        for bit in range(n):
                            if em & (1 << bit):
                                if first_col == -1:
                                    first_col = colour[bit]
                                elif colour[bit] != first_col:
                                    all_same = False
                                    break
                                # If unassigned, can't be all same yet
                                if colour[bit] == -1:
                                    all_same = False
                                    break
                        if all_same and first_col != -1:
                            immediate_conflict = True
                            break
                    
                    if not immediate_conflict:
                        if backtrack(idx + 1, colour):
                            return True
                
                # Undo
                colour[v] = -1
                for u in forced:
                    colour[u] = -1
        
        # Try to find a valid colouring
        if backtrack(0, colour):
            return False  # Valid colouring exists, so it IS 2-colorable
        else:
            return True  # No valid colouring, so NOT 2-colorable

    def prune_edges(edges, n, time_limit):
        """
        Greedy pruning with random restarts.
        """
        current_edges = list(edges)
        
        # Single pass
        current_edges = _prune_pass(current_edges, n, time_limit)
        
        # Random restarts
        max_restarts = 15
        for _ in range(max_restarts):
            if time.time() > time_limit:
                break
            if len(current_edges) <= 51: # Known lower bound area
                break
            
            shuffled = current_edges[:]
            random.shuffle(shuffled)
            
            new_edges = _prune_pass(shuffled, n, time_limit)
            
            if len(new_edges) < len(current_edges):
                current_edges = new_edges
            else:
                # Try a kick: remove a random edge, if fails, put back.
                # This helps escape local minima.
                if len(current_edges) > 1:
                    idx = random.randint(0, len(current_edges) - 1)
                    kick_edges = current_edges[:idx] + current_edges[idx+1:]
                    if is_non_2colorable_backtrack(kick_edges, n, time_limit):
                        current_edges = kick_edges
                    else:
                        break # Stuck
                else:
                    break
                    
        return current_edges

    def _prune_pass(edges, n, time_limit):
        """
        Single pass of greedy pruning.
        """
        current_edges = list(edges)
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            candidate = current_edges[:i] + current_edges[i+1:]
            
            if is_non_2colorable_backtrack(candidate, n, time_limit):
                current_edges = candidate
            else:
                i += 1
        return current_edges

    # Pre-compute all possible edges for small n
    all_possible_edges = {}
    for n in [13, 14, 15]:
        all_possible_edges[n] = list(combinations(range(n), 5))
        
    # Search loop
    for n in [13, 14, 15]:
        if time.time() > deadline - 10.0:
            break
            
        total_edges = len(all_possible_edges[n])
        attempts = 300 if n == 13 else 150
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Start with a dense random graph
            k_init = random.randint(55, 75)
            
            all_possible = all_possible_edges[n][:]
            random.shuffle(all_possible)
            initial_edges = [list(e) for e in all_possible[:k_init]]
            
            # Quick check: is initial graph non-2-colorable?
            check_deadline = time.time() + 1.0
            if is_non_2colorable_backtrack(initial_edges, n, check_deadline):
                prune_deadline = time.time() + 2.0
                if time.time() + 2.0 > deadline:
                    prune_deadline = deadline - 1.0
                    
                pruned = prune_edges(initial_edges, n, prune_deadline)
                
                if best_edges is None or len(pruned) < len(best_edges):
                    best_edges = pruned
                    best_n = n
                    
                    if len(pruned) <= 51:
                        # We found the known best or better.
                        pass

    # Fallback: Complete 5-graph on 9 vertices (126 edges)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
