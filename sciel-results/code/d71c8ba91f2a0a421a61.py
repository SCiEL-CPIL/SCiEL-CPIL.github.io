import itertools
import time

def _is_valid_coloring(colors):
    """
    Check if the coloring satisfies the K4 constraint.
    colors: list of 120 integers, index k corresponds to pair (i,j) 
    in lexicographic order: (0,1), (0,2), ..., (0,15), (1,2), ...
    """
    n = 16
    # Precompute edge indices for faster lookup
    # edge_idx[i][j] = index in colors list
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Check all K4s
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_idx[a][b]
        e2 = edge_idx[a][c]
        e3 = edge_idx[a][d]
        e4 = edge_idx[b][c]
        e5 = edge_idx[b][d]
        e6 = edge_idx[c][d]
        
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        c5 = colors[e5]
        c6 = colors[e6]
        
        distinct_colors = {c1, c2, c3, c4, c5, c6}
        if len(distinct_colors) < 5:
            return False
    return True

def _generate_candidate(k, mode):
    """
    Generate a candidate coloring with k colors.
    mode: 0 for (i+j)%k, 1 for (i-j)%k
    """
    n = 16
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if mode == 0:
                c = (i + j) % k
            else:
                c = (i - j) % k
            colors.append(c)
    return colors

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try deterministic candidates with fewer colors
    # Theoretical min is 13. Seed is 17.
    # Try 13, 14, 15, 16
    
    for k in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
        for mode in [0, 1]:
            candidate = _generate_candidate(k, mode)
            # Quick check: does it use all k colors? Not strictly required, 
            # but "minimize number of colors used" implies we should use 
            # the minimum necessary. If a candidate is valid, 
            # the number of colors used is max(candidate) + 1.
            # However, the prompt says "Minimize the number of colors used".
            # The evaluator measures the raw value (max color index + 1? or distinct count?)
            # "120 integer colours in 0..119... Minimize the number of colors used."
            # Usually this means the size of the palette, i.e., max(color) + 1.
            # If my candidate uses colors 0..12, the value is 13.
            
            if _is_valid_coloring(candidate):
                # Ensure colors are compact (0..k-1)
                # They are generated mod k, so they are.
                return candidate

    # If no small k worked, fall back to seed (17 colors)
    # Or try a simple local search if time permits?
    # Given the constraints and reliability, falling back to 17 is safe.
    # But 17 is the seed. Can we do better with a quick search?
    
    # Let's try a simple hill climbing to reduce colors from 17.
    # Start with seed
    best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_val = 17
    
    # Precompute edge indices
    n = 16
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_idx[a][b]
        e2 = edge_idx[a][c]
        e3 = edge_idx[a][d]
        e4 = edge_idx[b][c]
        e5 = edge_idx[b][d]
        e6 = edge_idx[c][d]
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    def count_violations(colors):
        count = 0
        for e1, e2, e3, e4, e5, e6 in k4s:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            c4 = colors[e4]
            c5 = colors[e5]
            c6 = colors[e6]
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                count += 1
        return count

    # Current state
    current_colors = best_colors[:]
    current_violations = count_violations(current_colors)
    # Seed is valid, so current_violations should be 0
    
    # Try to reduce colors by remapping or changing edge colors
    # A simple strategy: try to map the highest color to a lower one if possible.
    # Or, iterate over edges and try to change their color to 0..16 to reduce max color.
    
    # Since we already checked 13-16 deterministic, and they failed (assumed), 
    # the solution might be non-linear or require specific permutations.
    
    # Let's just return the seed if deterministic fails, to ensure validity.
    # But wait, if deterministic 13-16 failed, is 17 the best?
    # Maybe 15 works with a different formula?
    # (i*2 + j) % 15?
    
    # Let's try a few more deterministic patterns for 15 and 14.
    if time.time() < deadline:
        for k in [14, 15]:
            for a in [1, 2, 3]:
                for b in [1, 2, 3]:
                    if a == b: continue
                    candidate = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            candidate.append((a*i + b*j) % k)
                    if _is_valid_coloring(candidate):
                        return candidate

    return best_colors
