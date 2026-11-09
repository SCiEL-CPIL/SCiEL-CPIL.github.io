"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute all 4-cycles: for each 4-vertex subset, there are 3 distinct 4-cycles
# A 4-cycle on vertices {a,b,c,d} consists of 4 edges forming a cycle.
# For vertices a<b<c<d, the 3 cycles are:
#   (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
#   (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
#   (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Then add offset for j within vertex i's edges
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as lists of 4 edge indices
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                _CYCLES.append([_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)])
                # Cycle 2: a-b-d-c-a
                _CYCLES.append([_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)])
                # Cycle 3: a-c-b-d-a
                _CYCLES.append([_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)])

NUM_CYCLES = len(_CYCLES)

def _count_colors(colors):
    """Count the number of distinct colors used."""
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def _find_violation(colors):
    """Return a violating cycle or None."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if len({c0, c1, c2, c3}) == 4:
            return cycle
    return None

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    best_colors = [0] * NUM_EDGES
    best_score = 1  # at least 1 color used
    
    # Start with a random valid-ish coloring and improve via local search
    # Strategy: use a structured coloring that tends to avoid 4 distinct colors on 4-cycles
    
    # Approach: Use a small number of colors initially, then try to increase
    # We'll use simulated annealing / local search
    
    # Initial coloring: assign colors based on a pattern
    # Try coloring by edge "distance" or some structure
    random.seed(42)
    
    # Start with all zeros (valid, 1 color)
    colors = [0] * NUM_EDGES
    current_score = 1
    
    # Local search: try to change an edge's color to a new unused color
    # If it creates a violation, try to fix it
    
    # Better approach: start with a coloring using k colors where k is small
    # and gradually increase
    
    # Let's try: color edges by (i+j) mod k for some k
    # This tends to create structure that avoids 4 distinct colors
    
    for k in range(2, 120):
        if time.time() > deadline:
            break
        
        # Try coloring: color(i,j) = (i + j) % k
        test_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = _edge_idx(i, j)
                test_colors[idx] = (i + j) % k
        
        if _is_valid(test_colors):
            score = _count_colors(test_colors)
            if score > best_score:
                best_score = score
                best_colors = test_colors[:]
        
        # Try coloring: color(i,j) = (i * j) % k
        test_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = _edge_idx(i, j)
                test_colors[idx] = (i * j) % k
        
        if _is_valid(test_colors):
            score = _count_colors(test_colors)
            if score > best_score:
                best_score = score
                best_colors = test_colors[:]
        
        # Try coloring: color(i,j) = (i - j) % k
        test_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = _edge_idx(i, j)
                test_colors[idx] = (i - j) % k
        
        if _is_valid(test_colors):
            score = _count_colors(test_colors)
            if score > best_score:
                best_score = score
                best_colors = test_colors[:]
    
    # Now do local search from the best coloring found
    if time.time() < deadline:
        colors = best_colors[:]
        current_score = best_score
        
        # Local search: try changing individual edges to new colors
        # and check validity
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            # Try random edge changes
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                edge_idx = random.randint(0, NUM_EDGES - 1)
                old_color = colors[edge_idx]
                
                # Try a new random color
                new_color = random.randint(0, 119)
                if new_color == old_color:
                    continue
                
                colors[edge_idx] = new_color
                
                # Check if still valid
                if _is_valid(colors):
                    new_score = _count_colors(colors)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                        improved = True
                    else:
                        colors[edge_idx] = old_color
                else:
                    colors[edge_idx] = old_color
            
            if not improved:
                no_improve_count += 1000
        
        if current_score > best_score:
            best_score = current_score
            best_colors = colors[:]
    
    return best_colors
# EVOLVE-BLOCK-END
