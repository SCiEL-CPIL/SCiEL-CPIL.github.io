"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Directions in the triangular lattice (axial coordinates)
    # (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
    # Note: In axial coords (q, r), the 6 neighbors of (q, r) are:
    # (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Start with the origin
    points = set([(0, 0)])
    # Candidates for the next point to add
    candidates = set()
    
    # Initialize candidates with neighbors of the origin
    for dq, dr in directions:
        candidates.add((dq, dr))
        
    # Greedy growth: always add the candidate that maximizes the number of
    # connections to existing points.
    while len(points) < 32:
        best_candidate = None
        max_connections = -1
        
        for c in candidates:
            # Count how many existing points are neighbors of c
            connections = 0
            for dq, dr in directions:
                if (c[0] + dq, c[1] + dr) in points:
                    connections += 1
            
            if connections > max_connections:
                max_connections = connections
                best_candidate = c
                
        # Add the best candidate
        if best_candidate is not None:
            points.add(best_candidate)
            candidates.remove(best_candidate)
            
            # Add neighbors of the new point to candidates
            for dq, dr in directions:
                new_cand = (best_candidate[0] + dq, best_candidate[1] + dr)
                if new_cand not in points:
                    candidates.add(new_cand)
                    
    # Convert set to list of lists as required
    # Sorting ensures deterministic output, though not strictly required by "distinct"
    result = sorted([list(p) for p in points])
    return result
# EVOLVE-BLOCK-END
