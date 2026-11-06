import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_used(colors):
    return len(set(colors))

def is_valid_coloring(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c1 = (colors[edge_idx(a,b)], colors[edge_idx(b,c)],
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)])
                    if len(set(c1)) == 4:
                        return False
                    c2 = (colors[edge_idx(a,b)], colors[edge_idx(b,d)],
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)])
                    if len(set(c2)) == 4:
                        return False
                    c3 = (colors[edge_idx(a,c)], colors[edge_idx(c,b)],
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)])
                    if len(set(c3)) == 4:
                        return False
    return True

def construct():
    start_time = time.time()
    
    # Strategy: Backtracking that builds coloring vertex by vertex.
    # Key insight: If an edge has a unique color, it can't be part of any
    # valid 4-cycle without creating a rainbow. So every color must appear
    # on at least 2 edges. We enforce this as a pruning condition.
    
    # We'll assign edges in order: (0,1), (0,2), ..., (0,15), (1,2), ...
    # When assigning edge (i,j) where j is the "new" vertex, we check:
    # 1. No rainbow 4-cycle is created
    # 2. Any color used exactly once so far must be "rescued" by this edge
    #    or we prune (unless the color appears on an edge that will be assigned later)
    
    # Actually, let's use a different structural approach:
    # Partition vertices into groups and use a group-based coloring scheme.
    
    # Approach: Use a "blow-up" of a small valid coloring.
    # If we have a valid coloring on k vertices with c colors, we can blow up
    # each vertex into a group of size ~N/k, and edges between groups get
    # the color of the corresponding edge in the base coloring.
    # Within groups, we need additional colors, but we can reuse colors
    # from other groups to maintain the no-rainbow property.
    
    # Let me try a direct randomized local search with a different objective:
    # Start from a structured initialization, then use simulated annealing
    # to maximize distinct colors while maintaining validity.
    
    # Better: Use a constraint-based backtracking with memoization.
    
    # Let's try: assign colors to edges one by one, but use the constraint
    # that for any 4 vertices, at most 3 distinct colors appear on their 6 edges
    # in any 4-cycle. We'll use a smarter ordering.
    
    # New structural approach: Use a "two-partition" structure.
    # Split 16 vertices into two sets A and B of size 8 each.
    # Edges within A use colors from palette P_A.
    # Edges within B use colors from palette P_B.
    # Edges between A and B use colors from palette P_AB.
    # A 4-cycle can have: 2 edges in A, 2 in B (uses P_A, P_B colors)
    #                      1 in A, 2 cross, 1 in B (uses P_A, P_AB, P_B)
    #                      4 cross edges (uses P_AB)
    # For the 2-2 case: need |P_A ∪ P_B| colors on the cycle, but only 2 from each.
    # For the 1-2-1 case: 3 colors potentially.
    # For the 4-cross case: all from P_AB.
    
    # The critical constraint is the 1-2-1 case: if we have colors a (in A), 
    # x,y (cross), b (in B), we need a,x,y,b not all distinct.
    # This means for each pair (a,b) with a in P_A, b in P_B, the cross edges
    # connecting the specific vertices must use colors from {a,b} or repeat.
    
    # This is getting complex. Let me try a simpler structural approach:
    # Use a "star" decomposition. Pick a hub vertex 0. Edges from 0 to others
    # use distinct colors. All other edges use colors from the set of hub-edge
    # colors, creating a dependency structure.
    
    # Actually, let me try the most promising structural change:
    # A proper edge-coloring approach where we think in terms of 1-factors.
    # K_16 has 15 perfect matchings (1-factors). If we color each 1-factor
    # with a single color, we get 15 colors and no rainbow C4 (any C4 uses
    # at most 3 different 1-factors since it has 4 edges and each 1-factor
    # contributes at most 1 edge to a C4... actually that's not right).
    
    # Let me think more carefully. A C4 has 4 edges. In K_16, these 4 edges
    # belong to 4 different 1-factors (since no two edges in a C4 are in the
    # same 1-factor... actually that's not true either).
    
    # OK, let me just implement a fast randomized local search with a 
    # different initialization and move structure.
    
    # Initialize with a structured coloring:
    # Color edge (i,j) with (i+j) % 16, then try to improve.
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = (i + j) % 16
    
    # Verify initial coloring
    if not is_valid_coloring(colors):
        # Fallback to min(i,j)
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
    
    best_colors = list(colors)
    best_count = count_used(best_colors)
    
    # Local search: repeatedly pick a random edge and try to recolor it
    # to a new color (one not currently used) if it maintains validity.
    # Also try swapping colors between two edges.
    
    rng = random.Random(42)
    
    def try_recolor_to_new(colors, edge_i, edge_j, new_color):
        """Check if recoloring edge (edge_i, edge_j) to new_color is valid."""
        idx = edge_idx(edge_i, edge_j)
        old_color = colors[idx]
        if old_color == new_color:
            return False
        
        colors[idx] = new_color
        valid = is_valid_coloring(colors)
        colors[idx] = old_color
        return valid
    
    def try_swap(colors, idx1, idx2):
        """Check if swapping colors at idx1 and idx2 is valid."""
        if colors[idx1] == colors[idx2]:
            return False
        c1, c2 = colors[idx1], colors[idx2]
        colors[idx1] = c2
        colors[idx2] = c1
        valid = is_valid_coloring(colors)
        colors[idx1] = c1
        colors[idx2] = c2
        return valid
    
    # Precompute edge list
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    # Fast validity check: only check 4-cycles affected by a single edge change
    def is_valid_after_recolor(colors, ei, ej, new_color):
        """Check validity after recoloring edge (ei,ej) to new_color."""
        # Only 4-cycles containing edge (ei,ej) need checking
        # A 4-cycle containing (ei,ej) has form: ei-ej-x-y-ei
        # where x,y are other vertices, and the cycle is ei-ej-x-y-ei or ei-ej-y-x-ei
        # For each pair (x,y) of other vertices, there are specific 4-cycles.
        
        other = [v for v in range(N) if v != ei and v != ej]
        for xi in range(len(other)):
            for xj in range(xi + 1, len(other)):
                x = other[xi]
                y = other[xj]
                # 4 vertices: ei, ej, x, y
                # The 4-cycles on these 4 vertices:
                # 1. ei-ej-x-y-ei: edges (ei,ej), (ej,x), (x,y), (y,ei)
                # 2. ei-ej-y-x-ei: edges (ei,ej), (ej,y), (y,x), (x,ei)
                # 3. ei-x-ej-y-ei: edges (ei,x), (x,ej), (ej,y), (y,ei) - doesn't use (ei,ej) directly
                #    Wait, this uses (ej,y) and (ei,x), not (ei,ej).
                
                # So cycles 1 and 2 use edge (ei,ej).
                
                # Cycle 1: ei-ej-x-y-ei
                c1 = (new_color, colors[edge_idx(ej, x)], colors[edge_idx(x, y)], colors[edge_idx(y, ei)])
                if len(set(c1)) == 4:
                    return False
                
                # Cycle 2: ei-ej-y-x-ei
                c2 = (new_color, colors[edge_idx(ej, y)], colors[edge_idx(y, x)], colors[edge_idx(x, ei)])
                if len(set(c2)) == 4:
                    return False
        return True
    
    # Main local search loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < DEADLINE - 5:
        if no_improve_count > max_no_improve:
            # Try a different random restart
            rng.seed(int(time.time() * 1000) % 100000)
            no_improve_count = 0
            # Perturb: randomly recolor 5 edges
            for _ in range(5):
                ei, ej = rng.choice(edges)
                c = rng.randint(0, 119)
                if is_valid_after_recolor(best_colors, ei, ej, c):
                    best_colors[edge_idx(ei, ej)] = c
            continue
        
        ei, ej = rng.choice(edges)
        idx = edge_idx(ei, ej)
        old_color = best_colors[idx]
        
        # Try to assign a new color (not currently used)
        used = set(best_colors)
        new_colors = [c for c in range(120) if c not in used]
        
        if new_colors:
            new_color = rng.choice(new_colors)
            if is_valid_after_recolor(best_colors, ei, ej, new_color):
                best_colors[idx] = new_color
                no_improve_count = 0
                continue
        
        # Try to recolor to an existing color (might free up a color elsewhere)
        # Or try a swap with another edge
        if rng.random() < 0.3:
            # Try swap
            idx2 = rng.randint(0, NUM_EDGES - 1)
            if idx2 != idx and best_colors[idx] != best_colors[idx2]:
                # Find vertices for idx2
                # This is slow; skip for now
                pass
        
        # Try random recolor to any color
        c = rng.randint(0, 119)
        if c != old_color and is_valid_after_recolor(best_colors, ei, ej, c):
            best_colors[idx] = c
            # Check if this freed up a color
            if old_color not in best_colors:
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Final verification
    if not is_valid_coloring(best_colors):
        # Fallback
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
