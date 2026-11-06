"""
Combines the robust K32 seed with a vertex-growth strategy.
Instead of searching for a valid coloring of K_N from scratch or via random init,
we start with the known valid K32 coloring and attempt to add vertices one by one.
For each new vertex, we solve a Constraint Satisfaction Problem (CSP) to assign
colors to the edges connecting the new vertex to existing vertices, ensuring
no monochromatic triangles are formed.
"""
import time
import random

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _mul(a, b):
    """Multiply in GF(16) with primitive polynomial 0b10011."""
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index in flat colors list for pair (i, j) with i < j."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _try_add_vertex(colors, n, rng):
    """
    Attempts to add a new vertex (index n) to the existing K_n coloring.
    Returns the new colors list if successful, None otherwise.
    
    Constraints:
    Let c_v be the color of edge (v, n) for v in 0..n-1.
    For any i < j < n, we must NOT have (c_i == c_j == colors[i, j]).
    """
    m = n * (n - 1) // 2
    new_colors = colors[:] + [0] * n  # Pre-allocate space for new edges
    
    # We need to assign colors to edges (0, n), (1, n), ..., (n-1, n).
    # These correspond to indices in the new flat array.
    # Index of (v, n) in the new array of size (n+1):
    # _color_index(n+1, v, n)
    
    # Backtracking search for the assignment of c_0, ..., c_{n-1}
    assignment = [-1] * n
    valid = [True] * n  # Track which colors are still possible for each vertex? 
                         # Actually, we just do DFS with constraint checking.
    
    # To speed up, we can order variables. 
    # Let's just use simple backtracking with random ordering if needed, 
    # or standard 0..n-1 order.
    
    # Optimization: Pre-compute constraints?
    # For a fixed v, if we assign color c to edge (v, n), 
    # then for all u < v, if colors[u, v] == c, then edge (u, n) cannot be c.
    
    def dfs(v):
        if v == n:
            return True
        
        # Try colors 0..3 for edge (v, n)
        # Shuffle colors to add randomness
        cols = [0, 1, 2, 3]
        rng.shuffle(cols)
        
        for c in cols:
            # Check if assigning color c to (v, n) violates any constraint with u < v
            conflict = False
            for u in range(v):
                # If edge (u, v) has color c, and edge (u, n) has color c, 
                # then triangle (u, v, n) is monochromatic.
                # We need to check if assignment[u] is already c.
                if assignment[u] == c:
                    # Check color of (u, v)
                    idx_uv = _color_index(n, u, v)
                    if colors[idx_uv] == c:
                        conflict = True
                        break
            
            if not conflict:
                assignment[v] = c
                if dfs(v + 1):
                    return True
                assignment[v] = -1
        
        return False

    if dfs(0):
        # Fill the new_colors array
        for v in range(n):
            idx = _color_index(n + 1, v, n)
            new_colors[idx] = assignment[v]
        return new_colors
    
    return None

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with the guaranteed valid K32
    current_colors = _seed_32()
    current_n = 32
    best_colors = current_colors[:]
    best_n = current_n
    
    # Try to grow
    max_target = 50  # Chung's record
    # We might not reach 50, but we should try up to a reasonable limit or time out
    
    # To handle potential backtracking explosions, we limit the number of 
    # random restarts for the CSP at each step.
    
    for target_n in range(33, max_target + 1):
        if _time_left() < 5.0:
            break
            
        success = False
        # Try multiple times with different random seeds for the CSP
        # The backtracking order is fixed (0 to n-1), but color choice is shuffled.
        # We can also try different vertex orderings if we implement a more complex CSP,
        # but simple shuffling of colors usually helps.
        
        attempts = 100
        for _ in range(attempts):
            if _time_left() < 1.0:
                break
            new_colors = _try_add_vertex(current_colors, current_n, rng)
            if new_colors is not None:
                current_colors = new_colors
                current_n += 1
                best_colors = current_colors[:]
                best_n = current_n
                success = True
                break
        
        if not success:
            # If we can't add the next vertex, we stop.
            # We could try to "repair" the current_n coloring to make it easier to grow,
            # but that risks breaking validity.
            break
            
    return {'n': best_n, 'colors': best_colors}
