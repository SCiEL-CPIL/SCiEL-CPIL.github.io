import time
import random
import itertools

def construct():
    deadline = time.time() + 110  # Leave 40s safety margin
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx

    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # Identify parallelogram K4s
    # A K4 {a,b,c,d} is a parallelogram in F2^4 if a+b+c+d = 0
    # We represent vertices as 4-bit integers
    # Check condition: (a ^ b ^ c ^ d) == 0
    parallelogram_k4s = []
    for k4_idx, quad in enumerate(k4s):
        a, b, c, d = quad
        if (a ^ b ^ c ^ d) == 0:
            parallelogram_k4s.append(k4_idx)
            
    # Initial coloring: Difference coloring
    # Color = (i ^ j) if i != j. Since i,j are 0..15, i^j is 1..15.
    # We use 0..14 as colors. So color = (i ^ j) - 1.
    # Wait, i^j can be 0 if i==j, but i!=j so i^j != 0.
    # Range of i^j is 1..15.
    # So colors are 0..14.
    colours = [0] * n_edges
    for idx, (i, j) in enumerate(pairs):
        colours[idx] = (i ^ j) - 1
        
    def count_colors_in_k4(k4_idx):
        edges = k4_edges[k4_idx]
        return len(set(colours[e] for e in edges))
        
    def is_valid():
        for k4_idx in range(len(k4s)):
            if count_colors_in_k4(k4_idx) < 5:
                return False
        return True
        
    # Local Search
    best_colours = colours[:]
    best_num_colors = len(set(colours))
    
    # We focus on fixing parallelogram K4s first, as they are the only violated ones initially
    # In diff coloring, parallelograms have 3 colors.
    # Non-parallelograms have 6 colors.
    
    max_iterations = 5000
    iterations = 0
    improved = True
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Find a violated K4
        violated_k4s = [k4_idx for k4_idx in range(len(k4s)) if count_colors_in_k4(k4_idx) < 5]
        
        if not violated_k4s:
            break
            
        # Pick a random violated K4
        k4_idx = random.choice(violated_k4s)
        edges = k4_edges[k4_idx]
        current_colors = set(colours[e] for e in edges)
        
        # Try to fix by changing one edge
        fixed = False
        for e_idx in edges:
            # Try all colors 0..14
            for c in range(15):
                if c in current_colors:
                    # Changing to an existing color doesn't increase the count
                    # But it might help if we are trying to reduce color usage?
                    # No, we need INCREASE count to >= 5.
                    # So we must pick a color NOT in current_colors.
                    continue
                # Check if this change helps
                # New set would be (current_colors - {colours[e_idx]}) | {c}
                # If colours[e_idx] is unique in the K4, size stays same or increases?
                # Actually, we just need the new size to be >= 5.
                
                # Optimization: only try if it increases the count
                # Or just try and verify?
                # Let's just try and verify the specific K4
                old_c = colours[e_idx]
                colours[e_idx] = c
                if count_colors_in_k4(k4_idx) >= 5:
                    # Check if this breaks other K4s?
                    # We will do a global check later or accept small breaks?
                    # Better to accept and let the loop fix others.
                    # But we should avoid breaking other K4s if possible.
                    # For speed, let's just accept and see if we converge.
                    improved = True
                    fixed = True
                    break
                colours[e_idx] = old_c
            if fixed:
                break
                
        if not fixed:
            # Try changing two edges (diagonals)
            # Identify diagonals in this parallelogram
            # In a parallelogram {a,b,c,d} with a^b^c^d=0,
            # diagonals are (a,c) and (b,d) if a+b=c+d?
            # Actually, any pairing works?
            # In F2^4, the 3 pairings of 4 vertices are:
            # (a,b)|(c,d), (a,c)|(b,d), (a,d)|(b,c)
            # One of these pairs has sum a+b=c+d?
            # a^b^c^d=0 => a^b = c^d.
            # So the edges (a,b) and (c,d) have the same difference vector.
            # In the difference coloring, they have the same color.
            # These are the "parallel" sides? No, in a parallelogram, opposite sides are parallel.
            # In F2^4, "parallel" means same difference.
            # So (a,b) and (c,d) are opposite sides.
            # The diagonals are the other two pairs.
            
            # Let's just pick any two edges that are not opposite sides?
            # Or simpler: just pick two random edges and change them to two new distinct colors.
            
            # Pick two edges
            e1, e2 = random.sample(edges, 2)
            
            # Find colors not in current_colors
            available = [c for c in range(15) if c not in current_colors]
            if len(available) >= 2:
                c1, c2 = random.sample(available, 2)
                old_c1, old_c2 = colours[e1], colours[e2]
                colours[e1] = c1
                colours[e2] = c2
                if count_colors_in_k4(k4_idx) >= 5:
                    improved = True
                else:
                    colours[e1] = old_c1
                    colours[e2] = old_c2

    # Final check and cleanup
    valid = is_valid()
    if valid:
        best_colours = colours[:]
        best_num_colors = len(set(colours))
    else:
        # If not valid, we might have made things worse.
        # Return the best valid solution found so far?
        # We didn't track history.
        # Let's do a final greedy repair or just return the initial if this is worse.
        # Actually, the loop only accepts changes that fix the current K4.
        # It might break others.
        # We should verify the final state.
        # If invalid, we can try to revert to the initial 15-color solution if it was valid?
        # The initial 15-color solution is NOT valid (parallelograms have 3 colors).
        # So we MUST fix them.
        
        # If we fail to find a valid solution, we fall back to a known valid 17-color solution.
        # But we should try harder.
        # Let's run a few more iterations if time permits.
        pass

    # If still invalid, use the fallback 17-color solution
    if not is_valid():
        # Fallback: (i+j)%17
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return colours

# EVOLVE-BLOCK-END
