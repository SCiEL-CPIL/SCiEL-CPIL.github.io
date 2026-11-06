import itertools
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    n = 16
    
    # Precompute all 4-combinations
    quads = list(itertools.combinations(range(n), 4))
    
    # Precompute edge indices for each quad
    # Edge (i, j) with i < j.
    # We will use a 16x16 matrix for lookup.
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
    # Actually, simpler:
    # idx = 0
    # for i in range(n):
    #   for j in range(i+1, n):
    #     idx[i,j] = idx
    #     idx += 1
    
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
            
    def build_edge_matrix(coloring):
        """
        coloring: list of 120 ints
        returns: 16x16 numpy array
        """
        M = np.zeros((n, n), dtype=np.int32)
        for (i, j), c in zip([ (i,j) for i in range(n) for j in range(i+1, n) ], coloring):
            M[i, j] = c
            M[j, i] = c
        return M

    def is_valid_fast(M):
        """
        Check if every K4 in M has at least 5 distinct colors.
        M is a 16x16 numpy array.
        """
        for a, b, c, d in quads:
            c1 = int(M[a, b])
            c2 = int(M[a, c])
            c3 = int(M[a, d])
            c4 = int(M[b, c])
            c5 = int(M[b, d])
            c6 = int(M[c, d])
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                return False
        return True

    def make_coloring_formula(k):
        """
        Returns list of 120 colors for (i+j) % k
        """
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append((i + j) % k)
        return colors

    # Try simple formulas first
    for k in range(13, 18):
        if time.time() > deadline:
            break
        colors = make_coloring_formula(k)
        M = build_edge_matrix(colors)
        if is_valid_fast(M):
            return colors

    # If simple formulas failed, use Simulated Annealing
    # Start with a valid coloring (17 colors)
    current_coloring = make_coloring_formula(17)
    best_coloring = current_coloring[:]
    best_num_colors = 17
    
    # Precompute quad edges for fast update
    # quad_edges[quad_idx] = list of 6 edge indices
    quad_edges = []
    for a, b, c, d in quads:
        edges = [
            edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
            edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]
        ]
        quad_edges.append(edges)
        
    # edge_to_quads[edge_idx] = list of quad indices
    edge_to_quads = [[] for _ in range(120)]
    for q_idx, edges in enumerate(quad_edges):
        for e in edges:
            edge_to_quads[e].append(q_idx)

    def count_violations(coloring, M):
        """
        Returns number of violated quads.
        """
        viol = 0
        for a, b, c, d in quads:
            c1 = int(M[a, b])
            c2 = int(M[a, c])
            c3 = int(M[a, d])
            c4 = int(M[b, c])
            c5 = int(M[b, d])
            c6 = int(M[c, d])
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                viol += 1
        return viol

    def get_num_colors(coloring):
        return max(coloring) + 1 if coloring else 0

    # Initialize state
    M = build_edge_matrix(current_coloring)
    current_violations = count_violations(current_coloring, M)
    current_colors = get_num_colors(current_coloring)
    
    # Cost function: violations * 1000 + colors
    def cost(viol, cols):
        return viol * 1000 + cols

    current_cost = cost(current_violations, current_colors)
    best_cost = current_cost
    
    T = 50.0
    cooling_rate = 0.999
    
    max_steps = 2000000 # Adjust based on speed
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        # Pick random edge
        e_idx = random.randint(0, 119)
        old_color = current_coloring[e_idx]
        
        # Pick new color
        # Limit new color to current max color to encourage merging
        # Or allow new colors up to best_num_colors?
        # Let's pick from 0 to best_num_colors - 1
        new_color = random.randint(0, best_num_colors - 1)
        
        if old_color == new_color:
            continue
            
        # Calculate delta cost
        # Affected quads
        affected_quads = edge_to_quads[e_idx]
        
        old_viol_delta = 0
        new_viol_delta = 0
        
        for q_idx in affected_quads:
            a, b, c, d = quads[q_idx]
            
            # Get colors of the 6 edges
            # We need to update one of them
            # Identify which edge in the quad is e_idx
            # quad_edges[q_idx] has the 6 edge indices
            # We can just rebuild the set for the quad
            
            # Old state
            edges = quad_edges[q_idx]
            old_colors = [int(M[quads[q_idx][0], quads[q_idx][1]])] # This is slow
            
            # Faster: store the 6 colors in a list?
            # Let's just recalculate the distinct count for the quad
            
            # Get vertices
            v0, v1, v2, v3 = quads[q_idx]
            
            # Helper to get color from matrix or list
            def get_col(i, j):
                if i > j: i, j = j, i
                return int(M[i, j])
                
            old_c1 = get_col(v0, v1)
            old_c2 = get_col(v0, v2)
            old_c3 = get_col(v0, v3)
            old_c4 = get_col(v1, v2)
            old_c5 = get_col(v1, v3)
            old_c6 = get_col(v2, v3)
            
            old_distinct = len({old_c1, old_c2, old_c3, old_c4, old_c5, old_c6})
            old_viol = 1 if old_distinct < 5 else 0
            
            # New state
            # Update M temporarily? No, just check
            # If e_idx is one of the edges, its color changes to new_color
            # We need to know if e_idx is in this quad.
            # It is, by definition of affected_quads.
            
            # Which edge is it?
            # We can compare indices
            # But we already have the colors.
            # If old_color was one of the colors, it changes.
            
            # Let's create a new set of colors
            new_colors = {old_c1, old_c2, old_c3, old_c4, old_c5, old_c6}
            # Remove old_color if it's unique? No, just replace.
            # This is tricky if multiple edges have the same color.
            
            # Easier: Update M, check, revert M.
            # Since we are in a tight loop, M update is fast.
            
            # Find which edge in the quad is e_idx
            # quad_edges[q_idx] is list of edge indices
            # We don't have a direct mapping from edge index to position in quad edges list easily without search.
            # But we can just update M[v0,v1] etc based on which pair matches e_idx.
            
            # Let's identify the pair
            # e_idx corresponds to (i,j)
            # We can precompute edge_to_pair
            # edge_pairs[e_idx] = (i,j)
            
            # Let's add this precomputation
            # Actually, we can just check all 6 pairs
            # This is 6 checks per quad. 91 quads per edge. 546 checks.
            # Acceptable.
            
            # Update M
            # We need to know which M entry to change.
            # Let's precompute edge_to_vertices
            # edge_vertices[e_idx] = (i, j)
            
            # I'll add this to the main block below.
            
            # For now, let's use a simpler SA that doesn't do incremental updates but full check if few steps.
            # Or just do full check every 100 steps? No, too slow.
            
            # Let's restart the SA loop with proper precomputation.
            
            pass # Break to write proper code

    # Proper Precomputation
    edge_vertices = []
    for i in range(n):
        for j in range(i+1, n):
            edge_vertices.append((i, j))
            
    # Re-run SA with incremental updates
    M = build_edge_matrix(current_coloring)
    current_violations = count_violations(current_coloring, M)
    current_colors = get_num_colors(current_coloring)
    current_cost = cost(current_violations, current_colors)
    
    T = 50.0
    cooling_rate = 0.999
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        e_idx = random.randint(0, 119)
        old_color = current_coloring[e_idx]
        new_color = random.randint(0, best_num_colors - 1)
        
        if old_color == new_color:
            continue
            
        affected_quads = edge_to_quads[e_idx]
        
        old_viol_delta = 0
        new_viol_delta = 0
        
        # We need to update M for the specific edge
        i, j = edge_vertices[e_idx]
        
        # Save old color in M
        # M[i,j] and M[j,i]
        
        for q_idx in affected_quads:
            v0, v1, v2, v3 = quads[q_idx]
            
            # Get 6 colors
            c1 = int(M[v0, v1])
            c2 = int(M[v0, v2])
            c3 = int(M[v0, v3])
            c4 = int(M[v1, v2])
            c5 = int(M[v1, v3])
            c6 = int(M[v2, v3])
            
            old_set = {c1, c2, c3, c4, c5, c6}
            old_distinct = len(old_set)
            old_viol = 1 if old_distinct < 5 else 0
            
            # Determine new colors
            # If (v0,v1) is the edge, c1 changes
            # We can check if (v0,v1) == (i,j) or (j,i)
            
            new_c1 = c1
            new_c2 = c2
            new_c3 = c3
            new_c4 = c4
            new_c5 = c5
            new_c6 = c6
            
            if (v0 == i and v1 == j) or (v0 == j and v1 == i):
                new_c1 = new_color
            elif (v0 == i and v2 == j) or (v0 == j and v2 == i):
                new_c2 = new_color
            elif (v0 == i and v3 == j) or (v0 == j and v3 == i):
                new_c3 = new_color
            elif (v1 == i and v2 == j) or (v1 == j and v2 == i):
                new_c4 = new_color
            elif (v1 == i and v3 == j) or (v1 == j and v3 == i):
                new_c5 = new_color
            elif (v2 == i and v3 == j) or (v2 == j and v3 == i):
                new_c6 = new_color
            
            new_set = {new_c1, new_c2, new_c3, new_c4, new_c5, new_c6}
            new_distinct = len(new_set)
            new_viol = 1 if new_distinct < 5 else 0
            
            old_viol_delta += old_viol
            new_viol_delta += new_viol
            
        # Update M
        M[i, j] = new_color
        M[j, i] = new_color
        current_coloring[e_idx] = new_color
        
        # Update color count
        # This is hard to do incrementally without a counter array.
        # Let's maintain a color_count array.
        
        # I need to add color_count to the state.
        
        # Let's refactor to include color_count.
        
        pass # Break

    # Refactored SA with color_count
    color_count = [0] * 20 # Max 17-19 colors
    for c in current_coloring:
        color_count[c] += 1
        
    def get_num_colors_from_count(counts):
        return sum(1 for c in counts if c > 0)

    current_colors = get_num_colors_from_count(color_count)
    current_cost = cost(current_violations, current_colors)
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        e_idx = random.randint(0, 119)
        old_color = current_coloring[e_idx]
        new_color = random.randint(0, best_num_colors - 1)
        
        if old_color == new_color:
            continue
            
        affected_quads = edge_to_quads[e_idx]
        
        old_viol_delta = 0
        new_viol_delta = 0
        
        i, j = edge_vertices[e_idx]
        
        for q_idx in affected_quads:
            v0, v1, v2, v3 = quads[q_idx]
            
            c1 = int(M[v0, v1])
            c2 = int(M[v0, v2])
            c3 = int(M[v0, v3])
            c4 = int(M[v1, v2])
            c5 = int(M[v1, v3])
            c6 = int(M[v2, v3])
            
            old_set = {c1, c2, c3, c4, c5, c6}
            old_distinct = len(old_set)
            old_viol = 1 if old_distinct < 5 else 0
            
            new_c1, new_c2, new_c3, new_c4, new_c5, new_c6 = c1, c2, c3, c4, c5, c6
            
            if (v0 == i and v1 == j) or (v0 == j and v1 == i):
                new_c1 = new_color
            elif (v0 == i and v2 == j) or (v0 == j and v2 == i):
                new_c2 = new_color
            elif (v0 == i and v3 == j) or (v0 == j and v3 == i):
                new_c3 = new_color
            elif (v1 == i and v2 == j) or (v1 == j and v2 == i):
                new_c4 = new_color
            elif (v1 == i and v3 == j) or (v1 == j and v3 == i):
                new_c5 = new_color
            elif (v2 == i and v3 == j) or (v2 == j and v3 == i):
                new_c6 = new_color
            
            new_set = {new_c1, new_c2, new_c3, new_c4, new_c5, new_c6}
            new_distinct = len(new_set)
            new_viol = 1 if new_distinct < 5 else 0
            
            old_viol_delta += old_viol
            new_viol_delta += new_viol
            
        # Calculate new violations
        new_violations = current_violations - old_viol_delta + new_viol_delta
        
        # Update color count
        color_count[old_color] -= 1
        color_count[new_color] += 1
        new_colors = get_num_colors_from_count(color_count)
        
        new_cost = cost(new_violations, new_colors)
        delta_cost = new_cost - current_cost
        
        # Acceptance
        if delta_cost < 0 or (T > 0 and random.random() < np.exp(-delta_cost / T)):
            # Accept
            current_violations = new_violations
            current_colors = new_colors
            current_cost = new_cost
            
            # Update M and coloring
            M[i, j] = new_color
            M[j, i] = new_color
            current_coloring[e_idx] = new_color
            
            # Update best
            if new_violations == 0 and new_colors < best_num_colors:
                best_num_colors = new_colors
                best_coloring = current_coloring[:]
                # Also update best_cost?
                # We want to minimize colors among valid ones.
                
        else:
            # Reject
            # Revert color_count
            color_count[old_color] += 1
            color_count[new_color] -= 1
            # M and coloring not changed
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Final polish: Greedy merge on best valid solution
    if best_violations == 0: # We need to track best_violations
        pass # The code above doesn't track best_violations explicitly, but best_coloring is only updated if valid.
    
    # Return best valid coloring
    return best_coloring

# The code above has a bug: best_violations is not tracked, and the initial best_coloring might not be valid if simple formulas failed?
# No, simple formulas are checked. If they fail, we start SA from 17-color valid solution.
# So best_coloring is always valid.

# Let's clean up and write the final code.
