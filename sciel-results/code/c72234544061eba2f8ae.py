import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 16
    # Initialize edge colours
    # We start with k=15 colours. If we find a valid one, we try to reduce.
    # If not, we might try k=16, but the goal is to minimize.
    # Let's try to find a valid 15-colouring first.
    
    best_cost = float('inf')
    best_colours = None
    current_k = 15
    
    # Precompute all K4s for quick access? 
    # Or just iterate over pairs for local moves.
    
    # Initialize random colouring with current_k colours
    # Edges are (i, j) with i < j
    # Map (i, j) to index for array storage
    def edge_index(i, j):
        return i * n + j # Simple enough, or use a dict. Array of size n*n is fine.
    
    # Actually, let's use a 2D array for colours, col[i][j]
    # Only i < j is used.
    
    def random_coloring(k):
        col = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                col[i][j] = random.randint(0, k - 1)
                col[j][i] = col[i][j]
        return col

    def cost_of_k4(col, u, v, w, x):
        # Check if K4 on u,v,w,x has >= 5 colours
        c1 = col[u][v]
        c2 = col[u][w]
        c3 = col[u][x]
        c4 = col[v][w]
        c5 = col[v][x]
        c6 = col[w][x]
        return len({c1, c2, c3, c4, c5, c6}) < 5

    def total_cost(col):
        cnt = 0
        for quad in itertools.combinations(range(n), 4):
            if cost_of_k4(col, *quad):
                cnt += 1
        return cnt

    # Local search
    # We will run local search to minimize the number of violating K4s.
    
    # Start with k=15
    k = 15
    col = random_coloring(k)
    current_cost = total_cost(col)
    
    if current_cost == 0:
        best_colours = col
        best_cost = 0
    else:
        # Local search
        # Precompute list of K4s containing each edge?
        # For edge (i, j), the K4s are (i, j, a, b) where a, b in {0..15} \ {i, j} and a < b.
        # There are 91 such K4s.
        
        # To speed up, we can precompute for each edge the list of K4 indices or tuples.
        edge_k4s = {}
        for i in range(n):
            for j in range(i + 1, n):
                others = [x for x in range(n) if x != i and x != j]
                k4s = []
                for a, b in itertools.combinations(others, 2):
                    # The K4 is (i, j, a, b). Order doesn't matter for cost, 
                    # but we need to evaluate cost_of_k4 correctly.
                    # cost_of_k4 takes 4 vertices.
                    k4s.append((i, j, a, b))
                edge_k4s[(i, j)] = k4s
        
        # Also, we need to update cost efficiently.
        # Instead of recalculating total cost, we can track the cost of each K4.
        # k4_cost[k4_tuple] = 1 if invalid, 0 if valid.
        # total_cost = sum(k4_cost.values())
        
        # Let's use a dictionary for k4 validity
        # Key: tuple of 4 sorted vertices
        k4_validity = {}
        for quad in itertools.combinations(range(n), 4):
            k4_validity[quad] = not cost_of_k4(col, *quad) # True if valid (>=5 colours)
        
        current_cost = sum(1 for v in k4_validity.values() if not v)
        
        if current_cost == 0:
            best_colours = col
            best_cost = 0
        
        iterations = 0
        while time.time() < deadline - 5.0 and best_cost > 0:
            iterations += 1
            # Pick a random edge
            i, j = random.sample(range(n), 2)
            if i > j: i, j = j, i
            
            # Pick a new colour
            new_c = random.randint(0, k - 1)
            old_c = col[i][j]
            
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            # Only K4s containing (i, j) are affected.
            delta = 0
            affected_k4s = edge_k4s[(i, j)]
            
            # We need to know the state of these K4s before and after change.
            # Before:
            # After:
            
            # To avoid modifying col prematurely, we simulate.
            # But cost_of_k4 depends on 6 edges.
            # For a K4 (i, j, a, b), the edges are (i,j), (i,a), (i,b), (j,a), (j,b), (a,b).
            # Only (i,j) changes.
            
            for quad in affected_k4s:
                u, v, w, x = quad
                # Ensure (i,j) is in the quad. It is by construction.
                # Check validity before
                was_valid = k4_validity[quad]
                
                # Check validity after
                # We can compute the set of colours in the K4 after change.
                # The colours are:
                # col[u][v], col[u][w], col[u][x], col[v][w], col[v][x], col[w][x]
                # One of these is col[i][j] which becomes new_c.
                
                # Let's just recompute the set of colours for this specific K4.
                # It's cheap (6 lookups, set creation).
                c1 = col[u][v]
                c2 = col[u][w]
                c3 = col[u][x]
                c4 = col[v][w]
                c5 = col[v][x]
                c6 = col[w][x]
                
                # Find which one is (i,j)
                # This is inefficient. Let's just construct the set with new_c.
                # Actually, we can just check if the set of colours changes size.
                # Old set: S_old
                # New set: S_new = (S_old - {old_c}) union {new_c}
                # This is tricky if old_c appears multiple times.
                
                # Simpler: Just calculate the new set of colours for this K4.
                # We need to map the edge (i,j) to its position in the 6 edges.
                # This is getting complex.
                
                # Alternative: Just recompute cost_of_k4 with a temporary change?
                # No, we can just check the condition:
                # Is len({c1, c2, c3, c4, c5, c6}) < 5 ?
                
                # Let's optimize the "was_valid" check by storing the set of colours?
                # No, storing sets for 1820 K4s is fine.
                
                # Let's restart the loop logic with a simpler, slightly slower but correct approach.
                # Given 1820 K4s, and 91 per edge, and 150s, we can afford some overhead.
                
                pass # Placeholder for logic below

        # Let's implement a cleaner local search.
        
        # State: col (2D list)
        # Cost: Number of invalid K4s.
        
        # We'll just use a simple hill climbing with random restarts if stuck.
        
        def get_k4_colours(col, u, v, w, x):
            return {col[u][v], col[u][w], col[u][x], col[v][w], col[v][x], col[w][x]}
        
        # Precompute K4s for each edge
        edge_k4s = {}
        for i in range(n):
            for j in range(i + 1, n):
                others = [x for x in range(n) if x != i and x != j]
                k4s = []
                for a, b in itertools.combinations(others, 2):
                    # Sort the quad for consistent key? Not needed if we use tuple of 4 specific vertices.
                    # But cost_of_k4 is symmetric.
                    k4s.append((i, j, a, b))
                edge_k4s[(i, j)] = k4s
                
        # Initial state
        col = random_coloring(k)
        
        # Calculate initial cost and store validity
        k4_valid = {}
        current_cost = 0
        for quad in itertools.combinations(range(n), 4):
            valid = len(get_k4_colours(col, *quad)) >= 5
            k4_valid[quad] = valid
            if not valid:
                current_cost += 1
                
        if current_cost == 0:
            best_colours = col
            best_cost = 0
        else:
            best_colours = col
            best_cost = current_cost
            
            # Local search
            no_improve_count = 0
            max_no_improve = 1000
            
            while time.time() < deadline - 5.0:
                if no_improve_count > max_no_improve:
                    # Restart with new random coloring
                    col = random_coloring(k)
                    k4_valid = {}
                    current_cost = 0
                    for quad in itertools.combinations(range(n), 4):
                        valid = len(get_k4_colours(col, *quad)) >= 5
                        k4_valid[quad] = valid
                        if not valid:
                            current_cost += 1
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = [row[:] for row in col]
                    if current_cost == 0:
                        break
                    continue
                
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                
                new_c = random.randint(0, k - 1)
                old_c = col[i][j]
                
                if new_c == old_c:
                    continue
                
                # Calculate delta
                delta = 0
                for quad in edge_k4s[(i, j)]:
                    u, v, w, x = quad
                    was_valid = k4_valid[quad]
                    
                    # Compute new validity
                    # The set of colours changes.
                    # Old colours: S_old
                    # New colours: S_new
                    # We can compute S_new by taking S_old, removing old_c (if it's the only one), adding new_c.
                    # This is error-prone if old_c appears multiple times.
                    # Just recompute the set.
                    
                    # To speed up, we can check if the size changes.
                    # S_old size is known? No, we only store boolean.
                    # Let's store the set of colours for each K4?
                    # 1820 sets of size <= 6. Memory is fine.
                    # Let's modify state to store k4_colours_set.
                    
                    pass # This approach is getting messy. Let's simplify.

        # Simplified approach:
        # Just recompute total cost from scratch every N steps? No, too slow.
        # Let's stick to the delta calculation but do it carefully.
        
        # We will store k4_colour_sets to make delta calculation easy.
        
        col = random_coloring(k)
        k4_colour_sets = {}
        for quad in itertools.combinations(range(n), 4):
            k4_colour_sets[quad] = get_k4_colours(col, *quad)
        
        def is_valid_set(s):
            return len(s) >= 5
            
        current_cost = sum(1 for s in k4_colour_sets.values() if not is_valid_set(s))
        
        if current_cost == 0:
            best_colours = col
            best_cost = 0
        else:
            best_colours = col
            best_cost = current_cost
            
            no_improve = 0
            while time.time() < deadline - 5.0:
                if no_improve > 5000:
                    # Restart
                    col = random_coloring(k)
                    for quad in itertools.combinations(range(n), 4):
                        k4_colour_sets[quad] = get_k4_colours(col, *quad)
                    current_cost = sum(1 for s in k4_colour_sets.values() if not is_valid_set(s))
                    no_improve = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = [row[:] for row in col]
                    if current_cost == 0:
                        break
                    continue

                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                
                new_c = random.randint(0, k - 1)
                old_c = col[i][j]
                
                if new_c == old_c:
                    continue
                
                delta = 0
                for quad in edge_k4s[(i, j)]:
                    u, v, w, x = quad
                    old_set = k4_colour_sets[quad]
                    was_valid = is_valid_set(old_set)
                    
                    # Construct new set
                    # We need to know if old_c was in the set.
                    # If old_c is in old_set, and new_c is not in (old_set - {old_c}), then size might change.
                    # Actually, new_set = (old_set - {old_c}) | {new_c}
                    # This is only true if old_c appears exactly once in the 6 edges of the K4.
                    # If old_c appears multiple times, removing it from the set doesn't remove it if it's still there.
                    # So:
                    # new_set = set()
                    # for edge in edges_of_quad:
                    #   if edge == (i,j) or edge == (j,i):
                    #       new_set.add(new_c)
                    #   else:
                    #       new_set.add(col[edge[0]][edge[1]])
                    
                    # This is 6 iterations. 91 K4s * 6 = 546 operations per move.
                    # Very fast.
                    
                    new_set = set()
                    # The edges of the K4 (u,v,w,x) are:
                    # (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
                    # We need to check which one is (i,j).
                    
                    # Helper to add edge colour
                    def add_edge_colour(a, b):
                        if (a == i and b == j) or (a == j and b == i):
                            new_set.add(new_c)
                        else:
                            new_set.add(col[a][b])
                            
                    add_edge_colour(u, v)
                    add_edge_colour(u, w)
                    add_edge_colour(u, x)
                    add_edge_colour(v, w)
                    add_edge_colour(v, x)
                    add_edge_colour(w, x)
                    
                    is_new_valid = is_valid_set(new_set)
                    
                    if was_valid and not is_new_valid:
                        delta += 1
                    elif not was_valid and is_new_valid:
                        delta -= 1
                    
                    # Update the stored set (temporarily, for delta calc consistency if we accepted? 
                    # No, we calculate delta for all, then decide. 
                    # But the next K4 in the loop uses the OLD col, so old_set is correct for it.
                    # We only update col if we accept.
                # End for quad
                
                if delta < 0 or (delta == 0 and random.random() < 0.1):
                    # Accept
                    col[i][j] = new_c
                    col[j][i] = new_c
                    current_cost += delta
                    no_improve = 0
                    
                    # Update the stored sets for affected K4s
                    for quad in edge_k4s[(i, j)]:
                        u, v, w, x = quad
                        # Recalculate set? Or use the one we built?
                        # We didn't store the new_set from the loop.
                        # Let's just recompute it. It's cheap.
                        k4_colour_sets[quad] = get_k4_colours(col, u, v, w, x)
                        
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = [row[:] for row in col]
                        
                    if current_cost == 0:
                        break
                else:
                    no_improve += 1

    # If we found a valid 15-colouring, try to reduce to 14 or 13?
    # The problem asks to minimize colours.
    # If best_cost == 0, we have a valid 15-colouring.
    # Can we reduce it?
    # We can try to merge colours.
    # Pick two colours c1, c2. Try to map c2 to c1.
    # Check if validity is maintained.
    
    if best_cost == 0:
        # Try to reduce colours
        # Current k is 15.
        # We can try to remove colour 14, 13, etc.
        # Start from the highest colour index used.
        
        # Let's find the max colour used.
        max_c = max(max(row) for row in best_colours)
        
        # We can try to reduce max_c down to 0.
        # But we must maintain validity.
        
        # Let's try a simple reduction:
        # For c from max_c down to 1:
        #   Try to map c to c-1.
        #   If valid, accept.
        
        # This is a greedy reduction.
        
        reduced_col = [row[:] for row in best_colours]
        
        # We need to check validity quickly.
        # Let's use the same local search infrastructure but for colour reduction.
        # Or just check all K4s. 1820 checks is fast.
        
        def is_valid_colouring(col):
            for quad in itertools.combinations(range(n), 4):
                if len(get_k4_colours(col, *quad)) < 5:
                    return False
            return True
            
        # Try to reduce
        # We'll iterate from max_c down to 1.
        # For each c, try to map it to 0..c-1.
        
        # To be safe, we'll just try to map to c-1.
        for c in range(max_c, 0, -1):
            # Check if colour c is used
            used = False
            for i in range(n):
                for j in range(i+1, n):
                    if reduced_col[i][j] == c:
                        used = True
                        break
                if used: break
            if not used:
                continue
                
            # Try to map c to c-1
            # Create a temporary copy
            temp_col = [row[:] for row in reduced_col]
            for i in range(n):
                for j in range(i+1, n):
                    if temp_col[i][j] == c:
                        temp_col[i][j] = c - 1
                        temp_col[j][i] = c - 1
                        
            if is_valid_colouring(temp_col):
                reduced_col = temp_col
                # Update max_c? No, we just removed c.
                # We can continue the loop.
                
        # Now, the number of colours used is max(reduced_col) + 1
        final_colours = []
        for i in range(n):
            for j in range(i + 1, n):
                final_colours.append(reduced_col[i][j])
        
        # The score is the number of colours used.
        # The evaluator measures the raw value (lower is better).
        # The contract says: "Minimize the number of colours used."
        # And "A valid colouring certifies f(16) <= colours used."
        # The evaluator likely returns the number of distinct colours used.
        
        return final_colours
    else:
        # Return the best we found, even if invalid? 
        # "An invalid result ... scores 0".
        # So we must return a valid one if possible.
        # If we didn't find one with k=15, we should try k=16 or 17.
        # The seed program returns 17 colours.
        # If we fail to find 15, we should fall back to a known valid 17-colouring.
        
        # Let's implement the seed program's logic as fallback.
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# Run a quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Check validity
    n = 16
    col = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = res[idx]
            col[j][i] = res[idx]
            idx += 1
            
    valid = True
    for quad in itertools.combinations(range(n), 4):
        u,v,w,x = quad
        c = {col[u][v], col[u][w], col[u][x], col[v][w], col[v][x], col[w][x]}
        if len(c) < 5:
            valid = False
            break
    print("Valid:", valid)
    print("Colours used:", len(set(res)))
