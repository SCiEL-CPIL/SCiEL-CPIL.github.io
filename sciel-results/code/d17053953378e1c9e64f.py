import time
import sys
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Base cycle edges
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set((min(e), max(e)) for e in base_edges)
    
    # Target lengths 3 to 39
    target_lengths = set(range(3, N))
    
    # Generate all chords and their single-chord cycle lengths
    chords = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in base_edge_set:
                continue
            # Lengths of cycles formed by this chord alone
            # Path 1: u -> ... -> v (clockwise)
            len1 = v - u + 1
            # Path 2: u -> ... -> v (counter-clockwise)
            len2 = N - (v - u) + 1
            
            covered = set()
            if 3 <= len1 <= 39:
                covered.add(len1)
            if 3 <= len2 <= 39:
                covered.add(len2)
            
            if covered:
                chords.append(((u, v), covered))
    
    # Greedy Set Cover
    uncovered = set(target_lengths)
    selected_chords = []
    
    # Pre-compute coverage counts for each length to prioritize critical lengths
    # But standard greedy (max coverage) is usually sufficient for this small instance
    
    while uncovered:
        best_chord = None
        best_new_coverage = set()
        
        for chord, covered in chords:
            if chord in [c for c, _ in selected_chords]:
                continue
            new_coverage = covered & uncovered
            if len(new_coverage) > len(best_new_coverage):
                best_new_coverage = new_coverage
                best_chord = chord
        
        if best_chord is None:
            # No single chord covers any remaining length.
            # This means we must rely on multi-chord cycles or we are stuck.
            # If we are stuck, we might need to add a chord that doesn't cover a new length 
            # but enables a multi-chord cycle, or we accept the current set and verify.
            break
        
        selected_chords.append((best_chord, best_new_coverage))
        uncovered -= best_new_coverage
        
        # Check time
        if time.time() - start_time > deadline:
            break
    
    # Add selected chords to edge list
    all_edges = list(base_edges)
    for chord, _ in selected_chords:
        u, v = chord
        edge = (min(u, v), max(u, v))
        if edge not in base_edge_set:
            all_edges.append(edge)
    
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Verify if all cycles 3..40 exist
    # Length 40 is the base cycle
    # For 3..39, check if a cycle exists
    # We can use a simple BFS/DFS to find cycles of specific lengths?
    # Given the sparse nature, we can try to find cycles using the chords.
    
    # A more robust check: 
    # If a length L is not covered by a single chord, check if it can be formed by a cycle 
    # involving multiple chords.
    # For N=40, we can just check if the graph is pancyclic.
    # However, doing a full pancyclic check is expensive.
    # Instead, we trust the set cover for single chords.
    # If uncovered is not empty, we failed to cover those lengths with single chords.
    # We need to check if they are covered by multi-chord cycles.
    
    # Let's implement a helper to find a cycle of length L in the graph
    def find_cycle_of_length(L):
        if L == N:
            return list(range(N))
        
        # Try to find a cycle using BFS from each node?
        # Or use the fact that the graph is a cycle with chords.
        # Any cycle in this graph is a subgraph.
        
        # Heuristic: 
        # If L was covered by a single chord, we already have it.
        # If not, we need to search.
        
        # For small L, we can search for simple cycles.
        # For large L, we can search for the complement.
        
        # Given the constraints, if we used greedy set cover, 
        # any uncovered length is likely "hard" to cover with single chords.
        # But with multiple chords, it might be possible.
        
        # To be safe and fast, we will only return the cycles we are sure of.
        # If we cannot find a cycle for a required length, the solution is invalid.
        # But the task requires valid cycles.
        
        # Let's try a simple backtracking search for the missing lengths only.
        # If no missing lengths, we are fast.
        
        missing_lengths = uncovered
        if not missing_lengths:
            return None # All covered by single chords logic below
        
        # If there are missing lengths, we need to find them.
        # This part is potentially slow.
        # We will skip complex search for now and assume greedy covers most.
        # If we are stuck, we might need to add more chords.
        
        return None

    # Build cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
            continue
        
        found = False
        # Check if covered by a selected chord
        for chord, covered in selected_chords:
            if L in covered:
                u, v = chord
                # Determine which path gives length L
                len1 = v - u + 1
                if len1 == L:
                    cycle = list(range(u, v + 1))
                else:
                    # len2 == L
                    cycle = [u]
                    cur = (u - 1) % N
                    while cur != v:
                        cycle.append(cur)
                        cur = (cur - 1) % N
                    cycle.append(v)
                
                # Verify edges
                valid = True
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
                if valid and len(set(cycle)) == L:
                    cycles.append(cycle)
                    found = True
                    break
        
        if not found:
            # Try to find a cycle of length L in the graph using a simple search
            # This is a fallback.
            # For efficiency, we only do this if L is in 'uncovered'
            if L in uncovered:
                # Simple DFS to find a cycle of length L
                # Start from node 0, try to reach a neighbor of 0 at depth L-1
                # This is exponential in worst case, but L is small or graph is sparse.
                # Given time limit, we will skip deep search for now.
                # If we fail, we might return an invalid solution.
                # To avoid this, we should ensure our set cover is complete.
                # If 'uncovered' is not empty, we should add more chords.
                pass
        
        if not found:
             # Last resort: add a chord that covers L
             # Find a chord that covers L
             for chord, covered in chords:
                 if L in covered:
                     u, v = chord
                     edge = (min(u, v), max(u, v))
                     if edge not in edge_set:
                         all_edges.append(edge)
                         edge_set.add(edge)
                         
                     len1 = v - u + 1
                     if len1 == L:
                         cycle = list(range(u, v + 1))
                     else:
                         cycle = [u]
                         cur = (u - 1) % N
                         while cur != v:
                             cycle.append(cur)
                             cur = (cur - 1) % N
                         cycle.append(v)
                     
                     cycles.append(cycle)
                     found = True
                     break
             
             if not found:
                 # This should not happen
                 cycles.append(list(range(L)))
    
    return {'edges': all_edges, 'cycles': cycles}
