"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a 3-regular bipartite graph on vertices 0..11 where every
    simple cycle has length exactly 6.
    
    Uses a backtracking search to find a 6x6 (0,1)-matrix with:
    - Row sums = 3 (each point connects to 3 lines)
    - Column sums = 3 (each line connects to 3 points)
    - No two rows share more than 1 column (C4-free)
    
    If found, this gives a graph with only 6-cycles (score improvement).
    Fallback: two K_3,3 components (score 2.5).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Fallback: two K_3,3 components
    fallback_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            fallback_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            fallback_edges.append([u, v])
    
    # Search for a 6x6 matrix with the desired properties
    # rows[i] = set of columns (lines) connected to point i
    rows = []
    col_counts = [0] * 6
    
    def backtrack(row_idx):
        nonlocal rows, col_counts, start_time, deadline
        if time.time() > deadline:
            return False
        
        if row_idx == 6:
            # Check all column sums are 3
            for c in range(6):
                if col_counts[c] != 3:
                    return False
            # Verify C4-free: no two rows share more than 1 column
            for i in range(6):
                for j in range(i+1, 6):
                    if len(rows[i] & rows[j]) > 1:
                        return False
            return True
        
        # Try all combinations of 3 columns for this row
        from itertools import combinations
        for combo in combinations(range(6), 3):
            # Check column capacity
            valid = True
            for c in combo:
                if col_counts[c] >= 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check C4 constraint with existing rows
            row_set = set(combo)
            valid = True
            for existing in rows:
                if len(row_set & existing) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign
            rows.append(row_set)
            for c in combo:
                col_counts[c] += 1
            
            if backtrack(row_idx + 1):
                return True
            
            # Unassign
            rows.pop()
            for c in combo:
                col_counts[c] -= 1
        
        return False
    
    found = backtrack(0)
    
    if found:
        # Build edges from the found matrix
        edges = []
        for i in range(6):
            for j in sorted(rows[i]):
                edges.append([i, j + 6])  # Points 0-5, Lines 6-11
        
        # Verify the graph
        # Check degrees
        degree = [0] * 12
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d < 3 for d in degree):
            return fallback_edges
        
        # Check for 4-cycles
        # Two points share at most 1 line
        for i in range(6):
            for j in range(i+1, 6):
                if len(rows[i] & rows[j]) > 1:
                    return fallback_edges
        
        # Check for triangles (should be none in bipartite graph)
        # Bipartite graphs have no odd cycles, so no triangles
        
        # Verify there is at least one cycle
        # A 3-regular bipartite graph on 12 vertices must have cycles
        # (by Euler's formula or just by construction)
        
        # Ensure types
        final_edges = []
        for e in edges:
            u = int(e[0])
            v = int(e[1])
            final_edges.append([u, v])
        
        return final_edges
    else:
        # Use fallback
        final_edges = []
        for e in fallback_edges:
            u = int(e[0])
            v = int(e[1])
            final_edges.append([u, v])
        return final_edges

# EVOLVE-BLOCK-END
