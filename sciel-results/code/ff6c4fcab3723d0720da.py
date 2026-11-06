import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def popcount(x):
        return x.bit_count()
        
    def build_masks(colors, n):
        # colors is a list of length n*(n-1)/2
        # We need to map index to (i,j)
        # Or store as 2D array
        # For speed in search, 2D array of small ints is better
        # But bitmasks need to be updated.
        # Let's store colors in a 2D list colors[i][j]
        masks = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        return masks

    def count_violations(masks, n):
        """
        Counts the number of pairs (u, v, c) such that 
        popcount(masks[c][u] & masks[c][v]) >= 2.
        This counts the number of 'bad' vertex pairs in each color.
        """
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = masks[c][u] & masks[c][v]
                    if popcount(common) >= 2:
                        total += 1
        return total

    def check_valid(masks, n):
        return count_violations(masks, n) == 0

    def random_coloring(n):
        colors = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 4)
                colors[i][j] = c
                colors[j][i] = c
        return colors

    def local_search(n, max_iters, deadline, start_colors=None):
        if start_colors is None:
            colors = random_coloring(n)
        else:
            colors = [row[:] for row in start_colors]
            
        masks = build_masks(colors, n)
        current_viol = count_violations(masks, n)
        
        if current_viol == 0:
            return colors, True
            
        best_colors = [row[:] for row in colors]
        best_viol = current_viol
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            old_c = colors[u][v]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Calculate delta in violations
            # We only need to check pairs involving u or v
            # Violations involving u: (u, w) for w != u, v
            # Violations involving v: (v, w) for w != u, v
            # Violations between u and v: (u, v)
            
            def calc_local_viol(masks, u, v, n):
                viol = 0
                # Pairs (u, w)
                for c in range(5):
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        common = masks[c][u] & masks[c][w]
                        if popcount(common) >= 2:
                            viol += 1
                # Pairs (v, w)
                for c in range(5):
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        common = masks[c][v] & masks[c][w]
                        if popcount(common) >= 2:
                            viol += 1
                # Pair (u, v)
                for c in range(5):
                    common = masks[c][u] & masks[c][v]
                    if popcount(common) >= 2:
                        viol += 1
                return viol

            # Current local violation count (before change)
            # Note: This is an approximation of the global change if other parts are 0
            # But we track global best.
            
            # To be safe and fast, we can just do the swap, count global, and revert if bad?
            # Global count is O(n^2). n=26 -> 325 pairs * 5 colors = 1625 ops.
            # 100k iters * 1625 ops = 162M ops. Might be too slow in Python.
            
            # Let's optimize: Only recalculate global if local change looks promising?
            # Or just do local delta.
            
            # Let's try to compute delta precisely.
            # Violations affected are those where u or v is one of the two vertices in the pair.
            # Let V(u, v) be the set of colors c where popcount(masks[c][u] & masks[c][v]) >= 2.
            
            # This is getting complex. Let's stick to a simpler heuristic:
            # Change color, count global violations.
            # If n=26, global count is fast enough if optimized.
            
            # Optimization: Use a helper to count violations for a single color quickly?
            # No, let's just do the global count but break early if possible? No.
            
            # Let's try a different approach:
            # Keep a list of "bad edges".
            # If we have no bad edges, we are done.
            
            # For now, let's do the swap and global count.
            # 120s / 1625 ops/iter = 73k iters. Should be feasible.
            
            # Apply change
            masks[old_c][u] &= ~(1 << v)
            masks[old_c][v] &= ~(1 << u)
            masks[new_c][u] |= (1 << v)
            masks[new_c][v] |= (1 << u)
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            new_viol = count_violations(masks, n)
            
            delta = new_viol - current_viol
            
            if delta < 0 or (delta == 0 and random.random() < 0.1) or (delta > 0 and random.random() < pow(2.718, -delta / temp)):
                current_viol = new_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = [row[:] for row in colors]
                if current_viol == 0:
                    return best_colors, True
            else:
                # Revert
                masks[new_c][u] &= ~(1 << v)
                masks[new_c][v] &= ~(1 << u)
                masks[old_c][u] |= (1 << v)
                masks[old_c][v] |= (1 << u)
                colors[u][v] = old_c
                colors[v][u] = old_c
                
            temp *= cooling
            
        return best_colors, False

    # Try to find valid n=26
    n = 26
    found_26 = False
    valid_26_colors = None
    
    # Multiple restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        colors, valid = local_search(n, max_iters=5000, deadline=deadline)
        if valid:
            found_26 = True
            valid_26_colors = colors
            break
        else:
            # Keep the best one found so far?
            # local_search returns best_colors even if not valid.
            # We should track the best across restarts.
            # But local_search doesn't expose the best if not valid? 
            # It returns best_colors.
            if valid_26_colors is None or count_violations(build_masks(colors, n), n) < count_violations(build_masks(valid_26_colors, n), n):
                valid_26_colors = colors

    if found_26:
        # Try to extend to n=27
        n_new = 27
        # Initialize new vertex 26
        # Colors for edges (26, i) for i in 0..25
        # We can start with the valid 26 coloring and add edges randomly
        # Then run local search on the whole graph, but fixing the first 26*26/2 edges?
        # No, we can re-optimize the whole thing, but start from the valid 26 + random 26 edges.
        
        colors_27 = [row[:] for row in valid_26_colors]
        # Extend to 27x27
        new_colors = [[-1] * n_new for _ in range(n_new)]
        for i in range(26):
            for j in range(26):
                new_colors[i][j] = valid_26_colors[i][j]
        
        # Initialize edges to new vertex 26
        for i in range(26):
            c = random.randint(0, 4)
            new_colors[i][26] = c
            new_colors[26][i] = c
            
        # Run local search on n=27
        # To speed up, we can restrict moves to edges involving 26 initially?
        # Or just run full search.
        colors_27, valid_27 = local_search(n_new, max_iters=20000, deadline=deadline, start_colors=new_colors)
        
        if valid_27:
            # Return n=27
            result_colors = []
            for i in range(n_new):
                for j in range(i + 1, n_new):
                    result_colors.append(colors_27[i][j])
            return {'n': n_new, 'colors': result_colors}
            
    # If we didn't find 27, return the best 26 we found
    # If we didn't find a valid 26, we must return something valid.
    # The fallback is n=11 or a known small valid one.
    # But we likely found a valid 26.
    
    if valid_26_colors is not None:
        # Verify
        masks = build_masks(valid_26_colors, 26)
        if check_valid(masks, 26):
            result_colors = []
            for i in range(26):
                for j in range(i + 1, 26):
                    result_colors.append(valid_26_colors[i][j])
            return {'n': 26, 'colors': result_colors}
            
    # Fallback: n=11
    n = 11
    colors = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # Simple valid coloring for small n
            # e.g. random until valid
            pass
            
    # Generate a valid small one
    # For n=11, random search is very fast
    colors, valid = local_search(11, max_iters=1000, deadline=deadline)
    if valid:
        result_colors = []
        for i in range(11):
            for j in range(i + 1, 11):
                result_colors.append(colors[i][j])
        return {'n': 11, 'colors': result_colors}
        
    # Last resort: return n=1
    return {'n': 1, 'colors': []}

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
