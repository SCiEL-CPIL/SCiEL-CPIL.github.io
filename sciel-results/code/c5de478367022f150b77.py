"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _check_validity(colors, n=16, min_colors=5):
    """Check if the edge coloring satisfies the constraint."""
    # Build adjacency color matrix
    col = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = colors[idx]
            idx += 1
    
    # Check all K4s
    for a,b,c,d in itertools.combinations(range(n), 4):
        cset = {col[a][b], col[a][c], col[a][d], col[b][c], col[b][d], col[c][d]}
        if len(cset) < min_colors:
            return False
    return True

def _search_solution():
    """Search for a valid coloring with minimal number of colors."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 16
    # Precompute all K4s and their edge indices
    all_k4s = []
    for a,b,c,d in itertools.combinations(range(n), 4):
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        all_k4s.append(edges)
    
    # Try linear forms (ai + bj) % m for various m and coefficients
    best_colors = None
    best_num_colors = 17  # Default from seed
    
    # First, try the known good solution
    default_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    if _check_validity(default_colors):
        best_colors = default_colors
        best_num_colors = 17
    
    # Search for better solutions with smaller m
    for m in range(13, 17):
        if time.time() > deadline:
            break
        for a_coeff in range(1, m):
            for b_coeff in range(1, m):
                if time.time() > deadline:
                    break
                colors = [(a_coeff*i + b_coeff*j) % m for i in range(n) for j in range(i+1, n)]
                if _check_validity(colors):
                    num_used = len(set(colors))
                    if num_used < best_num_colors:
                        best_num_colors = num_used
                        best_colors = colors
                        # If we hit the lower bound, stop
                        if num_used <= 13:
                            return best_colors
    
    # Try more complex forms if time permits
    if time.time() < deadline - 30 and best_num_colors > 13:
        # Try (ai + bj + ci*j) % m
        for m in range(13, 17):
            if time.time() > deadline:
                break
            for a_coeff in range(1, m):
                for b_coeff in range(1, m):
                    for c_coeff in range(1, m):
                        if time.time() > deadline:
                            break
                        colors = [(a_coeff*i + b_coeff*j + c_coeff*i*j) % m 
                                 for i in range(n) for j in range(i+1, n)]
                        if _check_validity(colors):
                            num_used = len(set(colors))
                            if num_used < best_num_colors:
                                best_num_colors = num_used
                                best_colors = colors
                                if num_used <= 13:
                                    return best_colors
    
    return best_colors

def construct():
    result = _search_solution()
    if result is None:
        # Fallback to known valid solution
        result = [(i + j) % 17 for i in range(16) for j in range(i+1, 16)]
    return result
# EVOLVE-BLOCK-END
