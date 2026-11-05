import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy:
      1. Use vectorized NumPy check for 2-colorability for n <= 16.
      2. Use backtracking for n > 16.
      3. Greedy deletion from complete hypergraph with random restarts.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_edges = None
    best_n = 9
    best_count = float('inf')

    def is_non_2colorable_backtrack(edges, n, time_limit=None):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        Uses backtracking with unit propagation.
        """
        if not edges:
            return False

        edge_vertices = [list(e) for e in edges]
        colour = [-1] * n
        
        def backtrack():
            if time_limit and time.time() > time_limit:
                return False
            
            # Find first unassigned vertex
            v = -1
            for i in range(n):
                if colour[i] == -1:
                    v = i
                    break
            
            if v == -1:
                # All assigned, check if valid
                for ev in edge_vertices:
                    first_colour = colour[ev[0]]
                    mono = True
                    for bit in ev[1:]:
                        if colour[bit] != first_colour:
                            mono = False
                            break
                    if mono:
                        return False
                return True
            
            for c in [0, 1]:
                colour[v] = c
                
                forced = {}
                conflict = False
                for ev in edge_vertices:
                    count_c = 0
                    count_1c = 0
                    unassigned_in_edge = []
                    for bit in ev:
                        if colour[bit] == c:
                            count_c += 1
                        elif colour[bit] == 1 - c:
                            count_1c += 1
                        else:
                            unassigned_in_edge.append(bit)
                    
                    if count_c == 4 and len(unassigned_in_edge) == 1:
                        u = unassigned_in_edge[0]
                        if u not in forced:
                            forced[u] = 1 - c
                        elif forced[u] != 1 - c:
                            conflict = True
                            break
                    
                    if count_1c == 4 and len(unassigned_in_edge) == 1:
                        u = unassigned_in_edge[0]
                        if u not in forced:
                            forced[u] = c
                        elif forced[u] != c:
                            conflict = True
                            break
                
                if not conflict:
                    for u, fc in forced.items():
                        if colour[u] == -1:
                            colour[u] = fc
                        elif colour[u] != fc:
                            conflict = True
                            break
                
                if not conflict:
                    immediate_conflict = False
                    for ev in edge_vertices:
                        first_col = colour[ev[0]]
                        if first_col == -1:
                            continue
                        all_same = True
                        for bit in ev[1:]:
                            if colour[bit] != first_col:
                                all_same = False
                                break
                        if all_same:
                            immediate_conflict = True
                            break
                    
                    if not immediate_conflict:
                        if backtrack():
                            return True
                
                colour[v] = -1
                for u in forced:
                    colour[u] = -1

        if backtrack():
            return False
        else:
            return True

    def is_non_2colorable_fast(edges, n):
        """
        Fast check using NumPy for n <= 16.
        Returns True if NOT 2-colorable.
        """
        if n > 16:
            return is_non_2colorable_backtrack(edges, n)
        
        if not edges:
            return False

        # Generate all 2^n colorings
        bits = np.arange(2**n)
        powers = 1 << np.arange(n)
        # mask shape (2^n, n)
        mask = bits[:, None] & powers[None, :]
        colors = (mask > 0).astype(np.uint8)
        
        any_mono = np.zeros(2**n, dtype=bool)
        
        for e in edges:
            # e is a tuple/list of 5 vertices
            c = colors[:, list(e)]
            all_0 = np.all(c == 0, axis=1)
            all_1 = np.all(c == 1, axis=1)
            mono = all_0 | all_1
            any_mono |= mono
            # Early exit if all colorings are invalid? No, we need to know if ANY is valid.
            # If any_mono becomes all True, we can break early?
            # If all colorings have at least one mono edge, then no valid coloring exists.
            # So if np.all(any_mono) is True, we can break and return True (not 2-colorable).
            if np.all(any_mono):
                return True

        # If we finish the loop and there is at least one coloring with no mono edges, it is 2-colorable.
        # If all colorings have at least one mono edge, it is not 2-colorable.
        return np.all(any_mono)

    def greedy_prune(edges, n, time_limit):
        """
        Greedy deletion from a set of edges.
        """
        current_edges = list(edges)
        
        while time.time() < time_limit:
            removed_any = False
            random.shuffle(current_edges)
            
            for i in range(len(current_edges)):
                if time.time() > time_limit:
                    break
                
                # Try to remove edge i
                candidate = current_edges[:i] + current_edges[i+1:]
                
                if is_non_2colorable_fast(candidate, n):
                    current_edges = candidate
                    removed_any = True
                    # Reset index to 0 to restart scan? Or continue?
                    # Continuing is faster, but might miss edges that become removable later.
                    # However, shuffling handles the randomness.
                    # Let's break and restart loop to re-shuffle?
                    # No, just continue.
                else:
                    # Edge is critical, keep it
                    pass
            
            if not removed_any:
                break
        
        return current_edges

    # Search loop
    # Try small n first, as they might have fewer edges?
    # Actually, more vertices allow sparser graphs.
    # But checking is faster for smaller n.
    
    for n in [9, 10, 11, 12, 13, 14, 15, 16]:
        if time.time() > deadline - 5.0:
            break
            
        all_possible = list(combinations(range(n), 5))
        initial_edges = [list(e) for e in all_possible]
        
        # Check if initial is non-2-colorable
        if not is_non_2colorable_fast(initial_edges, n):
            continue
            
        # Perform greedy pruning with several restarts
        restarts = 10 if n <= 12 else 5
        
        for r in range(restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Start with a random subset of the complete graph?
            # Or full graph? Full graph is safer to start with.
            # But full graph has many edges.
            # Let's start with a random dense subset to speed up initial pruning?
            # No, start with full graph to ensure non-2-colorability.
            
            current = greedy_prune(initial_edges, n, deadline - 1.0)
            
            if len(current) < best_count:
                best_count = len(current)
                best_edges = current
                best_n = n
                
                # If we hit the theoretical lower bound or known best, stop?
                # Known best is 51.
                if best_count <= 51:
                    # Try to push lower?
                    # But 51 is the known minimum for m(5).
                    # We can stop early to save time, or keep searching for 50?
                    # The task says "Minimize".
                    # Let's keep searching a bit more.
                    pass

    # Fallback
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
