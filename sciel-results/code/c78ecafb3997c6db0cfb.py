"""Four-colourings of K_n without monochromatic triangles.

Structural change: vertex-based CSP backtracking with forward checking.
Instead of edge-by-edge backtracking (depth O(n^2)), we assign the entire
color pattern for a new vertex at once (depth O(n) per vertex). The constraint
is that for each color c, the c-neighbors of the new vertex form an
independent set in the existing color-c graph. We use MRV + forward checking.
"""

import time
import random
import sys

def construct():
    deadline = time.time() + 130

    def solve_vertex_csp(n_existing, edge_color, deadline, seed=None):
        """
        Solve CSP for new vertex v = n_existing.
        Variables: color[0], ..., color[n_existing-1], each in {0,1,2,3}.
        Constraint: for pair (a,b), if edge_color[min(a,b)][max(a,b)] = c,
        then NOT (color[a]=c AND color[b]=c).
        Returns assignment list or None.
        """
        if seed is not None:
            random.seed(seed)
        
        n = n_existing
        possible = [0b1111] * n
        assignment = [-1] * n
        nodes = [0]
        
        # Precompute conflict lists
        conflict = [[[] for _ in range(4)] for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = edge_color[a][b]
                conflict[a][c].append(b)
                conflict[b][c].append(a)
        
        def backtrack(assigned_count):
            nodes[0] += 1
            if nodes[0] % 500000 == 0 and time.time() > deadline:
                return False
            if time.time() > deadline:
                return False
            
            if assigned_count == n:
                return True
            
            # MRV heuristic
            best_var = -1
            best_cnt = 5
            for a in range(n):
                if assignment[a] == -1:
                    cnt = bin(possible[a]).count('1')
                    if cnt == 0:
                        return False
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best_var = a
                        if cnt <= 1:
                            break
            
            if best_var == -1:
                return True
            
            a = best_var
            avail = []
            for c in range(4):
                if possible[a] & (1 << c):
                    avail.append(c)
            
            random.shuffle(avail)
            
            for c in avail:
                assignment[a] = c
                possible[a] = (1 << c)
                
                saved = []
                valid = True
                for b in conflict[a][c]:
                    if assignment[b] != -1:
                        if assignment[b] == c:
                            valid = False
                            break
                        continue
                    if possible[b] & (1 << c):
                        saved.append(b)
                        possible[b] &= ~(1 << c)
                        if possible[b] == 0:
                            valid = False
                            break
                
                if valid:
                    if backtrack(assigned_count + 1):
                        return True
                
                assignment[a] = -1
                possible[a] = 0b1111
                for b in saved:
                    possible[b] |= (1 << c)
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    def verify_coloring(n, colors):
        """colors[i][j] for i<j. Check no mono-triangle."""
        for c in range(4):
            for a in range(n):
                for b in range(a + 1, n):
                    if colors[a][b] != c:
                        continue
                    for d in range(b + 1, n):
                        if colors[a][d] == c and colors[b][d] == c:
                            return False
        return True

    # Build up from n=1
    n = 1
    edge_color = [[-1]]
    
    random.seed(42)
    
    while n < 62 and time.time() < deadline:
        result = None
        for seed in range(10):
            if time.time() > deadline:
                break
            result = solve_vertex_csp(n, edge_color, deadline, seed=seed)
            if result is not None:
                break
        
        if result is None:
            result = solve_vertex_csp(n, edge_color, deadline)
        
        if result is None:
            # Local search perturbation
            found = False
            for attempt in range(50):
                if time.time() > deadline:
                    break
                perturbed = [row[:] for row in edge_color]
                for _ in range(3):
                    a = random.randint(0, n - 2)
                    b = random.randint(a + 1, n - 1)
                    perturbed[a][b] = random.randint(0, 3)
                
                if not verify_coloring(n, perturbed):
                    continue
                
                result = solve_vertex_csp(n, perturbed, deadline, seed=attempt)
                if result is not None:
                    edge_color = perturbed
                    found = True
                    break
            
            if not found:
                break
        
        new_n = n + 1
        new_edge_color = [[-1] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(i + 1, n):
                new_edge_color[i][j] = edge_color[i][j]
        for a in range(n):
            new_edge_color[a][new_n - 1] = result[a]
        
        edge_color = new_edge_color
        n = new_n
    
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            colors_list.append(edge_color[i][j])
    
    return {'n': n, 'colors': colors_list}
