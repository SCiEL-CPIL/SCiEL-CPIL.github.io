"""Improved edge-colouring of K16: every K4 sees >=5 colours."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _k4_min_colours(edge_colours):
    """Return the minimum number of distinct colours over all K4 subgraphs."""
    n = 16
    # Precompute: for each pair (i,j), store the colour
    # edge_colours is a 16x16 list where edge_colours[i][j] = colour of edge (i,j)
    min_c = 6  # can't be more than 6 in a K4
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        cols = {
            edge_colours[a][b], edge_colours[a][c], edge_colours[a][d],
            edge_colours[b][c], edge_colours[b][d], edge_colours[c][d]
        }
        if len(cols) < min_c:
            min_c = len(cols)
            if min_c <= 4:
                return min_c
    return min_c

def _try_additive_mod(m):
    """Try c(i,j) = (i+j) mod m. Returns number of colours used or None if invalid."""
    n = 16
    if m < 13:
        return None
    edge = [[0]*n for _ in range(n)]
    used = set()
    for i in range(n):
        for j in range(i+1, n):
            c = (i + j) % m
            edge[i][j] = c
            edge[j][i] = c
            used.add(c)
    # Check all K4s
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        cols = {
            edge[a][b], edge[a][c], edge[a][d],
            edge[b][c], edge[b][d], edge[c][d]
        }
        if len(cols) < 5:
            return None
    return len(used)

def _try_multiplicative_mod(m):
    """Try c(i,j) = (i*j) mod m with vertex labels 1..16 (or subset)."""
    n = 16
    if m < 13:
        return None
    edge = [[0]*n for _ in range(n)]
    used = set()
    # Assign vertex i -> (i+1) mod m, but ensure distinct
    # Actually just use vertices 1,2,...,16 mapped to values in Z_m
    # If m > 16, vertex i -> i+1
    # If m <= 16, we need 16 distinct values in Z_m - only possible if m >= 16
    if m >= 16:
        vals = list(range(1, m+1))[:16]  # 1..16
    else:
        return None
    for i in range(n):
        for j in range(i+1, n):
            c = (vals[i] * vals[j]) % m
            edge[i][j] = c
            edge[j][i] = c
            used.add(c)
    # Check all K4s
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        cols = {
            edge[a][b], edge[a][c], edge[a][d],
            edge[b][c], edge[b][d], edge[c][d]
        }
        if len(cols) < 5:
            return None
    return len(used)

def _try_linear_mod(m, a, b, c, d, e, f):
    """Try c(i,j) = (a*i + b*j + c*i*j + d) mod m."""
    n = 16
    if m < 13:
        return None
    edge = [[0]*n for _ in range(n)]
    used = set()
    for i in range(n):
        for j in range(i+1, n):
            col = (a*i + b*j + c*i*j + d) % m
            edge[i][j] = col
            edge[j][i] = col
            used.add(col)
    for combo in itertools.combinations(range(n), 4):
        a_, b_, c_, d_ = combo
        cols = {
            edge[a_][b_], edge[a_][c_], edge[a_][d_],
            edge[b_][c_], edge[b_][d_], edge[c_][d_]
        }
        if len(cols) < 5:
            return None
    return len(used)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # First, try the known good 17-colour solution
    base_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    base_colours = 17
    
    best_solution = base_solution
    best_colours = base_colours
    
    # Try additive mod m for various m
    for m in range(13, 18):
        if time.time() > deadline:
            break
        res = _try_additive_mod(m)
        if res is not None and res < best_colours:
            # Reconstruct
            n = 16
            best_solution = [(i + j) % m for i in range(n) for j in range(i + 1, n)]
            best_colours = res
    
    # Try multiplicative
    for m in range(13, 30):
        if time.time() > deadline:
            break
        res = _try_multiplicative_mod(m)
        if res is not None and res < best_colours:
            n = 16
            vals = list(range(1, m+1))[:16]
            best_solution = [(vals[i] * vals[j]) % m for i in range(n) for j in range(i + 1, n)]
            best_colours = res
    
    # Try some random linear forms
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 20000:
        attempts += 1
        m = random.randint(13, 25)
        a = random.randint(0, m-1)
        b = random.randint(0, m-1)
        c = random.randint(0, m-1)
        d = random.randint(0, m-1)
        res = _try_linear_mod(m, a, b, c, d, 0, 0)
        if res is not None and res < best_colours:
            n = 16
            best_solution = [(a*i + b*j + c*i*j + d) % m for i in range(n) for j in range(i + 1, n)]
            best_colours = res
    
    # Local search: try to reduce colours by remapping
    if best_colours > 13 and time.time() < deadline - 10:
        # Try greedy colour merging
        # Build edge matrix
        n = 16
        edge = [[0]*n for _ in range(n)]
        for idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(i+1, n)]):
            edge[i][j] = best_solution[idx]
            edge[j][i] = best_solution[idx]
        
        # Get list of colours used
        all_colours = sorted(set(edge[i][j] for i in range(n) for j in range(i+1, n)))
        
        # Try merging pairs of colours
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            current_colours = sorted(set(edge[i][j] for i in range(n) for j in range(i+1, n)))
            for c1, c2 in itertools.combinations(current_colours, 2):
                if time.time() > deadline - 5:
                    break
                # Try merging c2 -> c1
                # Check if valid
                valid = True
                for combo in itertools.combinations(range(n), 4):
                    a, b, c, d = combo
                    cols = set()
                    for x, y in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]:
                        col = edge[x][y]
                        if col == c2:
                            col = c1
                        cols.add(col)
                    if len(cols) < 5:
                        valid = False
                        break
                if valid:
                    # Merge
                    for i in range(n):
                        for j in range(i+1, n):
                            if edge[i][j] == c2:
                                edge[i][j] = c1
                                edge[j][i] = c1
                    changed = True
                    break
        
        if changed:
            # Reconstruct solution
            new_solution = [edge[i][j] for i in range(n) for j in range(i+1, n)]
            new_colours = len(set(new_solution))
            if new_colours < best_colours:
                best_solution = new_solution
                best_colours = new_colours
    
    # Final validation
    n = 16
    edge = [[0]*n for _ in range(n)]
    for idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(i+1, n)]):
        edge[i][j] = best_solution[idx]
        edge[j][i] = best_solution[idx]
    
    valid = True
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        cols = {
            edge[a][b], edge[a][c], edge[a][d],
            edge[b][c], edge[b][d], edge[c][d]
        }
        if len(cols) < 5:
            valid = False
            break
    
    if not valid:
        # Fallback to the safe 17-colour solution
        return base_solution
    
    return best_solution
# EVOLVE-BLOCK-END
