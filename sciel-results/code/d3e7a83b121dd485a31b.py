import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n = 20
    # color_matrix[i][j] = color of edge (i,j), -1 means no edge
    # Use 0..19 for colors, -1 for no edge
    color_matrix = np.full((n, n), -1, dtype=np.int8)
    np.fill_diagonal(color_matrix, -2)  # mark diagonal
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if color_matrix[i][j] >= 0:
                    edges.append([i, j, int(color_matrix[i][j])])
        return edges
    
    def is_valid_4cycle(a, b, c, d):
        """Check if 4-cycle a-b-c-d-a has 4 distinct colors.
        All 4 edges must exist."""
        c1 = color_matrix[a][b]
        c2 = color_matrix[b][c]
        c3 = color_matrix[c][d]
        c4 = color_matrix[d][a]
        if c1 < 0 or c2 < 0 or c3 < 0 or c4 < 0:
            return True  # not a complete 4-cycle, no constraint
        return len([c1, c2, c3, c4]) == 4
    
    def count_violations():
        """Count all 4-cycle violations."""
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                if color_matrix[a][b] < 0:
                    continue
                for c in range(b+1, n):
                    if color_matrix[b][c] < 0:
                        continue
                    for d in range(c+1, n):
                        if color_matrix[c][d] < 0:
                            continue
                        if color_matrix[d][a] < 0:
                            continue
                        if not is_valid_4cycle(a, b, c, d):
                            violations += 1
        return violations
    
    def check_all_valid():
        """Check if current coloring is fully valid."""
        for a in range(n):
            for b in range(a+1, n):
                if color_matrix[a][b] < 0:
                    continue
                for c in range(b+1, n):
                    if color_matrix[b][c] < 0:
                        continue
                    for d in range(c+1, n):
                        if color_matrix[c][d] < 0:
                            continue
                        if color_matrix[d][a] < 0:
                            continue
                        if not is_valid_4cycle(a, b, c, d):
                            return False
        return True
    
    def try_add_edge(u, v):
        """Try to add edge (u,v) with some color. Returns True if successful."""
        if color_matrix[u][v] >= 0:
            return True
        # Try each color
        for c in range(n):
            color_matrix[u][v] = c
            color_matrix[v][u] = c
            # Check if this creates any 4-cycle violation
            # A 4-cycle containing edge (u,v): u-v-x-y-u
            # x and y must be distinct from u,v and each other
            ok = True
            for x in range(n):
                if x == u or x == v:
                    continue
                if color_matrix[v][x] < 0:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if color_matrix[x][y] < 0:
                        continue
                    if color_matrix[y][u] < 0:
                        continue
                    # Check 4-cycle u-v-x-y-u
                    c1 = color_matrix[u][v]  # c
                    c2 = color_matrix[v][x]
                    c3 = color_matrix[x][y]
                    c4 = color_matrix[y][u]
                    if len([c1, c2, c3, c4]) != 4:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
            else:
                color_matrix[u][v] = -1
                color_matrix[v][u] = -1
        return False
    
    def try_remove_edge(u, v):
        color_matrix[u][v] = -1
        color_matrix[v][u] = -1
    
    def edge_count():
        return int(np.sum(color_matrix >= 0)) // 2
    
    # Strategy 1: Greedy construction
    # Start with a random permutation-based coloring
    # Use a structured approach: assign colors based on (i+j) mod n or similar
    
    best_edges = 0
    best_matrix = color_matrix.copy()
    
    # Try multiple random seeds for greedy construction
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        color_matrix.fill(-1)
        np.fill_diagonal(color_matrix, -2)
        
        # Try to build a dense graph greedily
        # Order of edge additions matters
        edge_order = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edge_order)
        
        for (u, v) in edge_order:
            if time.time() > deadline:
                break
            try_add_edge(u, v)
        
        if check_all_valid():
            ec = edge_count()
            if ec > best_edges:
                best_edges = ec
                best_matrix = color_matrix.copy()
        
        if time.time() > deadline:
            break
    
    # Try to improve by adding more edges
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if color_matrix[u][v] < 0:
                    if try_add_edge(u, v):
                        improved = True
                        ec = edge_count()
                        if ec > best_edges:
                            best_edges = ec
                            best_matrix = color_matrix.copy()
        if not improved:
            break
    
    # Local search: try recoloring edges to enable more additions
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try to find a recoloring that allows adding an edge
        added = False
        for u in range(n):
            for v in range(u+1, n):
                if color_matrix[u][v] >= 0:
                    continue
                # Try removing some edge and recoloring to allow (u,v)
                # For each edge in a violating 4-cycle, try recoloring
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if color_matrix[v][x] < 0:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if color_matrix[x][y] < 0 or color_matrix[y][u] < 0:
                            continue
                        # Check if u-v-x-y-u is a violating 4-cycle
                        c1 = color_matrix[u][v]  # -1, not set yet
                        # We're trying to add (u,v), so check what color would work
                        # The existing cycle v-x-y-u-v has colors:
                        c_vx = color_matrix[v][x]
                        c_xy = color_matrix[x][y]
                        c_yu = color_matrix[y][u]
                        # We need a color c for (u,v) such that {c, c_vx, c_xy, c_yu} are all distinct
                        used = {c_vx, c_xy, c_yu}
                        available = [c for c in range(n) if c not in used]
                        if not available:
                            continue
                        # Try each available color
                        for c in available:
                            # Check if adding (u,v) with color c creates ANY violation
                            # We already checked this 4-cycle, but need to check all
                            color_matrix[u][v] = c
                            color_matrix[v][u] = c
                            ok = True
                            for x2 in range(n):
                                if x2 == u or x2 == v:
                                    continue
                                if color_matrix[v][x2] < 0:
                                    continue
                                for y2 in range(n):
                                    if y2 == u or y2 == v or y2 == x2:
                                        continue
                                    if color_matrix[x2][y2] < 0:
                                        continue
                                    if color_matrix[y2][u] < 0:
                                        continue
                                    c1 = color_matrix[u][v]
                                    c2 = color_matrix[v][x2]
                                    c3 = color_matrix[x2][y2]
                                    c4 = color_matrix[y2][u]
                                    if len([c1, c2, c3, c4]) != 4:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                added = True
                                ec = edge_count()
                                if ec > best_edges:
                                    best_edges = ec
                                    best_matrix = color_matrix.copy()
                                break
                            else:
                                color_matrix[u][v] = -1
                                color_matrix[v][u] = -1
                        if added:
                            break
                    if added:
                        break
                if added:
                    break
            if added:
                break
        if not added:
            break
    
    # Use best matrix found
    if best_edges > edge_count():
        color_matrix = best_matrix
    
    if not check_all_valid():
        # Fallback: return a valid star
        color_matrix.fill(-1)
        np.fill_diagonal(color_matrix, -2)
        for v in range(1, n):
            color_matrix[0][v] = 0
            color_matrix[v][0] = 0
    
    return get_edges()
