import time
import sys
import random
import numpy as np

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Precompute edge indices
    edge_list = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_list.append((i, j))
    E = len(edge_list) # 120
    
    # Map (i, j) to index
    idx_map = {}
    for idx, (i, j) in enumerate(edge_list):
        idx_map[(i, j)] = idx
        if i > j:
            idx_map[(j, i)] = idx
        else:
            idx_map[(i, j)] = idx
            
    # Precompute all 4-cycles (sets of 4 edges)
    # A 4-cycle is defined by 4 distinct vertices a, b, c, d.
    # The edges are (a,b), (b,c), (c,d), (d,a) or similar permutations.
    # For a set of 4 vertices {u, v, w, x}, there are 3 ways to form a 4-cycle using 4 edges.
    # Actually, the problem says "all three cycles on every four-vertex subset are checked".
    # This implies that for any 4 vertices, we must ensure that NO 4-cycle formed by them has 4 distinct colors.
    # The 3 cycles on 4 vertices {a,b,c,d} correspond to the 3 perfect matchings of K4? No.
    # A 4-cycle in a graph on 4 vertices uses 4 edges.
    # In K4, there are 3 distinct 4-cycles (each omitting a different pair of opposite edges? No).
    # Let's list them for vertices 0,1,2,3.
    # Cycle 1: 0-1-2-3-0 (edges: (0,1), (1,2), (2,3), (3,0))
    # Cycle 2: 0-1-3-2-0 (edges: (0,1), (1,3), (3,2), (2,0))
    # Cycle 3: 0-2-1-3-0 (edges: (0,2), (2,1), (1,3), (3,0))
    # Yes, these are the 3 distinct 4-cycles in K4.
    
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Vertices are a, b, c, d
                    # Cycle 1: a-b-c-d-a
                    e1 = idx_map[(a, b)]
                    e2 = idx_map[(b, c)]
                    e3 = idx_map[(c, d)]
                    e4 = idx_map[(d, a)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = idx_map[(a, b)]
                    e2 = idx_map[(b, d)]
                    e3 = idx_map[(d, c)]
                    e4 = idx_map[(c, a)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = idx_map[(a, c)]
                    e2 = idx_map[(c, b)]
                    e3 = idx_map[(b, d)]
                    e4 = idx_map[(d, a)]
                    cycles.append((e1, e2, e3, e4))
    
    # Conflict Graph for "Unique Color" assignments
    # If we assign a unique new color to an edge, it conflicts with another edge
    # if they are part of the same 4-cycle.
    # Why? If edge e1 and e2 are in a 4-cycle, and we give them unique colors,
    # we must ensure the other two edges in that cycle don't also have unique colors 
    # (or at least don't complete the rainbow).
    # Wait, the constraint is "No four-cycle may have four distinct edge colors".
    # If we assign unique colors to a set of edges S, we must ensure that for every cycle C,
    # C does NOT contain 4 edges from S.
    # This is equivalent to saying that S is an independent set in a hypergraph where
    # hyperedges are the 4-cycles? No.
    # If S contains 4 edges from a cycle C, then those 4 edges have 4 distinct colors (since they are in S, they have unique colors).
    # So the condition is: No 4-cycle contains 4 edges from S.
    # This is NOT a standard graph independent set problem (where no edge is in S).
    # It is a 4-uniform hypergraph independent set (no hyperedge is fully contained in S).
    
    # However, the inspirations mention "conflict graph where nodes represent potential new color assignments... edges connect assignments that would create a rainbow 4-cycle if simultaneously active".
    # If we just pick a set S of edges to have unique colors, and the rest have color 0.
    # A rainbow cycle is formed if 4 edges in a cycle all have distinct colors.
    # If an edge is in S, it has a unique color. If not, it has color 0.
    # A cycle has 4 distinct colors iff it has 4 edges in S.
    # (If it has 3 in S and 1 not, colors are {u1, u2, u3, 0} -> 4 distinct. Wait.
    # If 3 edges are in S, they have 3 distinct colors. The 4th edge has color 0.
    # Are u1, u2, u3, 0 distinct? Yes, because u_i > 0.
    # So if a cycle has ANY 3 or 4 edges from S, it has 4 distinct colors?
    # Let's check:
    # Case 4 edges in S: colors c1, c2, c3, c4. All distinct. Violation.
    # Case 3 edges in S, 1 edge not in S: colors c1, c2, c3, 0. All distinct. Violation.
    # Case 2 edges in S, 2 edges not in S: colors c1, c2, 0, 0. Distinct colors: {c1, c2, 0}. Size 3. OK.
    # Case 1 edge in S, 3 edges not in S: colors c1, 0, 0, 0. Distinct colors: {c1, 0}. Size 2. OK.
    # Case 0 edges in S: colors 0, 0, 0, 0. Distinct colors: {0}. Size 1. OK.
    
    # So the constraint is: No 4-cycle may contain 3 or 4 edges from S.
    # This is a packing problem.
    
    # To maximize |S|, we want to pick as many edges as possible such that no 4-cycle has >= 3 edges from S.
    
    # Heuristic: Greedy with random ordering.
    # We want to find a large set S.
    # Conflict: If we add edge e to S, we must ensure no cycle already has 2 edges in S (because adding e would make it 3).
    # Also, we must ensure no cycle already has 3 edges in S (impossible if we maintain the invariant that no cycle has >= 3).
    
    # So, we can greedily add edges e to S if:
    # For every cycle C containing e, the number of edges in C that are already in S is < 2.
    
    # Let's implement this.
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid_s(S):
        # S is a set of edge indices
        # Check all cycles
        for c in cycles:
            count_in_S = 0
            for e in c:
                if e in S:
                    count_in_S += 1
            if count_in_S >= 3:
                return False
        return True
        
    def greedy_max_s(max_time=1.0):
        """
        Greedily construct a set S of edges such that no 4-cycle has >= 3 edges in S.
        """
        t_start = time.time()
        # Randomly order edges
        edges = list(range(E))
        random.shuffle(edges)
        
        S = set()
        # For efficiency, maintain count of S-edges per cycle?
        # There are 1820 cycles (C(16,4) * 3 = 1820 * 3 = 5460 cycles).
        # E = 120.
        # Checking all cycles for each edge is 120 * 5460 * 4 ~ 2.6M ops. Very fast.
        
        # To speed up, we can precompute for each edge, the list of cycles it belongs to.
        edge_to_cycles = [[] for _ in range(E)]
        for i, c in enumerate(cycles):
            for e in c:
                edge_to_cycles[e].append(i)
                
        # cycle_counts[i] = number of edges in cycle i that are in S
        cycle_counts = [0] * len(cycles)
        
        for e in edges:
            if time.time() - t_start > max_time:
                break
            # Check if adding e is safe
            # e is in cycles edge_to_cycles[e]
            # For each such cycle, if cycle_counts[cycle_id] >= 2, adding e makes it 3. Violation.
            safe = True
            for cid in edge_to_cycles[e]:
                if cycle_counts[cid] >= 2:
                    safe = False
                    break
            if safe:
                S.add(e)
                for cid in edge_to_cycles[e]:
                    cycle_counts[cid] += 1
                    
        return S

    def refine_s(S, max_time=5.0):
        """
        Local search to try to increase |S|.
        Try to add edges not in S.
        Try to swap edges?
        """
        t_start = time.time()
        S = set(S)
        
        # Precompute cycle counts
        cycle_counts = [0] * len(cycles)
        for e in S:
            for cid in edge_to_cycles[e]:
                cycle_counts[cid] += 1
                
        # List of edges not in S
        not_in_S = [e for e in range(E) if e not in S]
        random.shuffle(not_in_S)
        
        improved = True
        while improved and time.time() - t_start < max_time:
            improved = False
            random.shuffle(not_in_S)
            for e in not_in_S:
                if time.time() - t_start > max_time:
                    break
                # Check if we can add e
                safe = True
                for cid in edge_to_cycles[e]:
                    if cycle_counts[cid] >= 2:
                        safe = False
                        break
                if safe:
                    S.add(e)
                    for cid in edge_to_cycles[e]:
                        cycle_counts[cid] += 1
                    not_in_S.remove(e)
                    improved = True
                    # Break inner loop to restart shuffle? Or continue?
                    # Continuing is fine, but removing from list while iterating is bad.
                    # We break and restart the outer while loop to reshuffle.
                    break
        
        return S

    best_S = set()
    
    # Run multiple greedy instances
    while time.time() < deadline - 10.0:
        S = greedy_max_s(max_time=0.5)
        S = refine_s(S, max_time=1.0)
        
        if len(S) > len(best_S):
            best_S = S
            
    # Final refinement on best_S
    best_S = refine_s(best_S, max_time=10.0)
    
    # Construct colors
    # Edges in best_S get unique colors 1, 2, ...
    # Edges not in best_S get color 0.
    
    colors = [0] * E
    color_id = 1
    for e in best_S:
        colors[e] = color_id
        color_id += 1
        
    # Verify validity just in case
    # (Optional but good for debugging, skip for speed if confident)
    
    return colors

# Helper to get edge index in lexicographic order (i, j) with 0 <= i < j < N
# This is not used in the new strategy directly as we use edge_list indices,
# but the output must be in the specific lexicographic order defined by the task.
# The task says: "120 integer colors ... for the lexicographically ordered pairs (i,j)"
# My edge_list is constructed in lexicographic order:
# for i in range(N): for j in range(i+1, N):
# So index 0 is (0,1), index 1 is (0,2), ..., index N-2 is (0, N-1), index N-1 is (1,2), etc.
# This matches the required order.

# Keep the old function name if required? "Keep the function names and signatures of the seed program."
# Seed program had construct().
# It also had get_edge_idx inside. I removed it. That should be fine as long as construct works.

# One detail: The task says "Maximize the number of colors actually used".
# My strategy uses 1 (for 0) + |S| colors.
# So maximizing |S| maximizes colors.

# Is it possible to get more colors by using a more complex coloring where not all "new" colors are unique?
# No, the constraint is on distinct colors in a cycle.
# If we use a color twice, it doesn't help increase the count of *distinct* colors used in the whole graph,
# unless we are reusing a color to satisfy a constraint that would otherwise force a merge.
# But since we are starting from a valid state (all 0s) and adding unique colors,
# the only way to increase the count is to add more unique colors.
# The bottleneck is the constraint "no 4-cycle has 4 distinct colors".
# If we assign unique colors to S, the constraint is "no 4-cycle has >= 3 edges from S".
# If we allowed reusing colors, we might be able to pack more edges with "non-zero" colors?
# But the score is "number of colors actually used".
# If I use color 1 on 10 edges, that's 1 color.
# If I use color 1 on 1 edge, color 2 on 1 edge, ..., color 10 on 1 edge, that's 10 colors.
# So it is always better to use unique colors for the "active" edges, as long as it doesn't violate the constraint.
# The constraint violation happens when a cycle has 4 distinct colors.
# If I use unique colors, a cycle has 4 distinct colors iff it has 4 edges with unique colors (or 3 unique + 1 zero).
# If I reuse colors, I might avoid the "4 distinct" situation.
# Example: Cycle has edges e1, e2, e3, e4.
# If e1, e2, e3 have colors 1, 2, 3. e4 has color 1.
# Distinct colors: {1, 2, 3}. Size 3. Valid.
# In my "unique color" model, if e1, e2, e3 are in S, they have colors 1, 2, 3.
# e4 is not in S, so color 0.
# Distinct colors: {1, 2, 3, 0}. Size 4. Invalid.
# So my model is STRICTER than necessary?
# If I could assign color 1 to e4 as well, it would be valid.
# But then e4 is not "unique".
# Does this allow me to use MORE colors overall?
# Suppose I have a set of edges. I want to assign them colors to maximize the number of distinct colors.
# This is a graph coloring problem on the hypergraph of cycles?
# No, it's simpler.
# We want to maximize the number of distinct colors used in the whole graph.
# Let C be the set of colors used.
# For each color c in C, let S_c be the set of edges with color c.
# The constraint is: For every 4-cycle, the set of colors on its edges has size < 4.
# This implies that for every 4-cycle, at least two edges share a color.
#
# If we use k colors, can we always do better by using unique colors for a subset?
# Actually, if we use a color on multiple edges, it "wastes" the potential for that color to be distinct in other cycles?
# No, the goal is just the count of distinct colors in the entire graph.
# It is generally optimal to use as many unique colors as possible, because reusing a color doesn't increase the count.
# The only reason to reuse is if using unique colors forces a violation.
# But if we have a set of edges S that we want to have unique colors, the condition for validity is:
# For every cycle, it cannot have 4 edges from S.
# AND it cannot have 3 edges from S and 1 edge from T (where T is the set of edges with color 0)?
# Wait. If an edge is not in S, it has color 0.
# If a cycle has 3 edges from S (colors c1, c2, c3) and 1 edge from T (color 0), the colors are {c1, c2, c3, 0}.
# Since c1, c2, c3 are unique and non-zero, and 0 is distinct from them, we have 4 distinct colors.
# So the condition "No 4-cycle has 3 or 4 edges from S" is CORRECT for the case where all other edges are color 0.
#
# What if we use multiple colors for the "non-unique" edges?
# Suppose we partition edges into S_1, S_2, ..., S_k, where S_i all have color i.
# And S_0 has color 0.
# We want to maximize k+1 (or k if 0 is not counted? "number of colors actually used" includes 0 if used).
# If we use color 0, it counts.
# So we want to maximize the number of colors.
# Is it better to have 10 colors with 1 edge each, or 5 colors with 2 edges each?
# 10 > 5.
# So we want to maximize the number of color classes.
# This is equivalent to finding the maximum number of colors such that there exists a valid assignment.
#
# My current strategy:
# 1. Pick a set S of edges to have unique colors 1..|S|.
# 2. All other edges have color 0.
# 3. Maximize |S|.
# Total colors = |S| + 1.
#
# Is it possible to get MORE colors by splitting the "0" edges into multiple colors?
# Suppose we have the "0" edges. Can we assign some of them color 1, some color 2, etc.?
# If we assign an edge e not in S to color 1, we must ensure no cycle has 4 distinct colors.
# If a cycle has 3 edges from S (colors c1, c2, c3) and e (color 1), and 1 is not c1,c2,c3, then we have 4 distinct colors.
# So e can only take a color that is already present in the cycle, or the cycle must not have 3 unique colors from S.
#
# This suggests that the "unique color for S, 0 for rest" strategy might be suboptimal if we can "recolor" some of the 0-edges to existing colors to allow more edges to be in S?
# No, the constraint on S is independent of the colors of the other edges, EXCEPT for the "3 from S + 1 other" case.
# If the "other" edge has color 0, and 0 is distinct from S's colors, it violates.
# If the "other" edge has color c1 (same as one of S's edges), then the colors are {c1, c2, c3, c1} -> {c1, c2, c3}. Size 3. Valid.
#
# So, if we recolor some "0" edges to match colors of S, we might be able to ADD more edges to S?
# No, adding to S means giving them a NEW unique color.
#
# Let's stick to the simpler model first: Maximize |S| such that no cycle has >= 3 edges in S.
# This gives |S|+1 colors.
#
# The inspirations scored 19. My seed scored 15.
# The seed used a hierarchical partition.
# The inspirations used local search.
# My current implementation uses a greedy independent set on the "3-in-cycle" hypergraph.
# This should be able to find larger S than the hierarchical partition.
#
# Let's run it.

# Note: The deadline is 60s. I use 50s.
# The greedy loop runs for ~10s, refinement for ~10s.
# Total ~20-30s. Safe.

# One optimization: The cycle count check in `refine_s` can be optimized.
# But for N=16, it's small.

# Final check on the output format.
# "120 integer colors in 0..119"
# My colors are 0 and 1..|S|.
# |S| is at most 120.
# So colors are in 0..119. Correct.

# The function `construct` returns the list of colors.

