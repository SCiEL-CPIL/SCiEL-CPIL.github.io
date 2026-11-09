# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product
from collections import deque
import sys

def _is_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _solve_lp(sep_x, sep_y, n=10):
    """
    Solve LP for a given separation graph.
    sep_x: list of (i,j) meaning square i is LEFT of square j
    sep_y: list of (i,j) meaning square i is BELOW square j
    """
    def vx(i): return 3*i
    def vy(i): return 3*i + 1
    def vs(i): return 3*i + 2
    
    nvars = 3 * n
    eps = 1e-10
    
    c = np.zeros(nvars)
    for i in range(n):
        c[vs(i)] = -1.0
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(nvars); row[vx(i)] = 1.0; row[vs(i)] = 0.5
        A_ub.append(row); b_ub.append(1.0)
        row = np.zeros(nvars); row[vy(i)] = 1.0; row[vs(i)] = 0.5
        A_ub.append(row); b_ub.append(1.0)
        row = np.zeros(nvars); row[vx(i)] = -1.0; row[vs(i)] = 0.5
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(nvars); row[vy(i)] = -1.0; row[vs(i)] = 0.5
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(nvars); row[vs(i)] = -1.0
        A_ub.append(row); b_ub.append(-eps)
    
    for (i, j) in sep_x:
        row = np.zeros(nvars)
        row[vx(i)] = 1.0; row[vx(j)] = -1.0
        row[vs(i)] = 0.5; row[vs(j)] = 0.5
        A_ub.append(row); b_ub.append(0.0)
    
    for (i, j) in sep_y:
        row = np.zeros(nvars)
        row[vy(i)] = 1.0; row[vy(j)] = -1.0
        row[vs(i)] = 0.5; row[vs(j)] = 0.5
        A_ub.append(row); b_ub.append(0.0)
    
    if not A_ub:
        return None
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None)] * nvars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            centers = np.array([[x[vx(i)], x[vy(i)]] for i in range(n)])
            sides = np.array([x[vs(i)] for i in range(n)])
            return centers, sides
        else:
            return None
    except Exception:
        return None

def _topo_sort(adj, n):
    in_degree = [0] * n
    for i in range(n):
        for j in adj[i]:
            in_degree[j] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return count == n

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # ---- Strategy 1: Systematic row-based layouts ----
    # For each partition of 10 into rows, try all
    def try_row_layout(counts):
        sep_x = []
        sep_y = []
        row_sq = []
        idx = 0
        for cnt in counts:
            row_sq.append(list(range(idx, idx+cnt)))
            idx += cnt
        for row in row_sq:
            for i in range(len(row)-1):
                sep_x.append((row[i], row[i+1]))
        for r in range(len(row_sq)-1):
            for i in row_sq[r]:
                for j in row_sq[r+1]:
                    sep_y.append((i, j))
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # All partitions of 10 into rows (up to 5 rows for speed)
    partitions = []
    def gen_partitions(remaining, max_part, current, max_rows):
        if not remaining:
            if len(current) <= max_rows:
                partitions.append(list(current))
            return
        if len(current) >= max_rows:
            return
        for p in range(min(max_part, remaining), 0, -1):
            current.append(p)
            gen_partitions(remaining - p, p, current, max_rows)
            current.pop()
    
    gen_partitions(10, 10, [], 5)
    
    for counts in partitions:
        if time.time() > deadline - 60:
            break
        try_row_layout(counts)
    
    # ---- Strategy 2: Random separation graphs ----
    np.random.seed(42)
    max_attempts = 8000
    for attempt in range(max_attempts):
        if time.time() > deadline - 40:
            break
        
        sep_x = []
        sep_y = []
        
        for i in range(n):
            for j in range(i+1, n):
                r = np.random.random()
                if r < 0.22:
                    sep_x.append((i, j))
                elif r < 0.44:
                    sep_x.append((j, i))
                elif r < 0.66:
                    sep_y.append((i, j))
                elif r < 0.88:
                    sep_y.append((j, i))
        
        adj_x = [[] for _ in range(n)]
        for (i, j) in sep_x:
            adj_x[i].append(j)
        adj_y = [[] for _ in range(n)]
        for (i, j) in sep_y:
            adj_y[i].append(j)
        
        if not _topo_sort(adj_x, n) or not _topo_sort(adj_y, n):
            continue
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # ---- Strategy 3: Guided random search with structure ----
    np.random.seed(123)
    for _ in range(5000):
        if time.time() > deadline - 20:
            break
        
        # Create a random "grid-like" structure
        # Pick a random number of rows and columns
        n_rows = np.random.randint(2, 6)
        n_cols = np.random.randint(2, 6)
        
        # Assign squares to grid cells
        cell_assignment = np.zeros(n, dtype=int)
        for i in range(n):
            cell_assignment[i] = np.random.randint(0, n_rows * n_cols)
        
        # Build separation constraints based on grid positions
        sep_x = []
        sep_y = []
        
        for i in range(n):
            for j in range(i+1, n):
                ci = cell_assignment[i]
                cj = cell_assignment[j]
                ri, ci_col = divmod(ci, n_cols)
                rj, cj_col = divmod(cj, n_cols)
                
                if ri < rj:
                    sep_y.append((i, j))
                elif ri > rj:
                    sep_y.append((j, i))
                elif ci_col < cj_col:
                    sep_x.append((i, j))
                elif ci_col > cj_col:
                    sep_x.append((j, i))
                # Same cell: no constraint (they'll be separated by LP if possible)
        
        adj_x = [[] for _ in range(n)]
        for (i, j) in sep_x:
            adj_x[i].append(j)
        adj_y = [[] for _ in range(n)]
        for (i, j) in sep_y:
            adj_y[i].append(j)
        
        if not _topo_sort(adj_x, n) or not _topo_sort(adj_y, n):
            continue
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # ---- Strategy 4: Local improvement via re-solving with perturbed constraints ----
    if best_centers is not None and time.time() < deadline - 10:
        # Try to improve by adding/removing constraints
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            # Start from current best, try adding a random constraint
            sep_x = []
            sep_y = []
            
            # Infer current separations from positions
            for i in range(n):
                for j in range(i+1, n):
                    dx = best_centers[j, 0] - best_centers[i, 0]
                    dy = best_centers[j, 1] - best_centers[i, 1]
                    if dx > best_sides[i]/2 + best_sides[j]/2 + 1e-9:
                        sep_x.append((i, j))
                    elif dx < -(best_sides[i]/2 + best_sides[j]/2 + 1e-9):
                        sep_x.append((j, i))
                    elif dy > best_sides[i]/2 + best_sides[j]/2 + 1e-9:
                        sep_y.append((i, j))
                    elif dy < -(best_sides[i]/2 + best_sides[j]/2 + 1e-9):
                        sep_y.append((j, i))
            
            # Perturb: remove one random constraint and add a different one
            if sep_x and np.random.random() < 0.5:
                idx = np.random.randint(len(sep_x))
                sep_x.pop(idx)
            elif sep_y and np.random.random() < 0.5:
                idx = np.random.randint(len(sep_y))
                sep_y.pop(idx)
            
            # Add a random constraint
            i, j = np.random.randint(n, size=2)
            if i != j:
                if i > j:
                    i, j = j, i
                if np.random.random() < 0.5:
                    sep_x.append((i, j))
                else:
                    sep_y.append((i, j))
            
            adj_x = [[] for _ in range(n)]
            for (i, j) in sep_x:
                adj_x[i].append(j)
            adj_y = [[] for _ in range(n)]
            for (i, j) in sep_y:
                adj_y[i].append(j)
            
            if not _topo_sort(adj_x, n) or not _topo_sort(adj_y, n):
                continue
            
            result = _solve_lp(sep_x, sep_y, n)
            if result:
                update_best(result[0], result[1])
    
    if best_centers is None:
        c = np.array([[0.5, 0.5]] * n)
        s = np.array([0.1] * n)
        return c, s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
