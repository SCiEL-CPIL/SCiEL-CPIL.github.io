import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def paley_colors(n):
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % n in squares else 1)
        return colors

    def build_adj(n, colors):
        adj = [np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = True
                adj[c][j, i] = True
                idx += 1
        return adj

    def is_valid(n, colors):
        adj = build_adj(n, colors)
        for c in [0, 1]:
            A = adj[c]
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v
                    W = A[:, u] & A[:, v]
                    W[u] = False
                    W[v] = False
                    
                    # Get indices of vertices in W
                    verts = np.where(W)[0]
                    if len(verts) < 4:
                        continue
                    
                    # For each pair (x, y) in verts, check if they have >= 2 common neighbors in W
                    # We can vectorize this check for all pairs (x, y) at once?
                    # Let M be the subgraph induced by verts.
                    # M = A[np.ix_(verts, verts)]
                    # We need to know if there exists x, y in verts such that 
                    # sum(M[x, :] & W) >= 2 ? No.
                    # We need sum( (A[:, x] & A[:, y] & W) ) >= 2.
                    # Let's iterate over pairs in verts. len(verts) is at most n-2=28.
                    # 28*27/2 = 378 pairs.
                    # For each pair, we do a bitwise and and count.
                    # This is fast in NumPy.
                    
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            common = A[:, x] & A[:, y] & W
                            if np.sum(common) >= 2:
                                return False
        return True

    def count_conflicts(n, colors):
        adj = build_adj(n, colors)
        conflicts = 0
        for c in [0, 1]:
            A = adj[c]
            for u in range(n):
                for v in range(u + 1, n):
                    W = A[:, u] & A[:, v]
                    W[u] = False
                    W[v] = False
                    verts = np.where(W)[0]
                    if len(verts) < 4:
                        continue
                    
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            common = A[:, x] & A[:, y] & W
                            bc = np.sum(common)
                            if bc >= 2:
                                conflicts += bc * (bc - 1) // 2
        return conflicts

    # Step 1: Check Paley(29)
    colors_29 = paley_colors(29)
    best_result = None
    if colors_29 and is_valid(29, colors_29):
        best_result = {'n': 29, 'colors': colors_29}

    # Step 2: Try to find n=30
    if time.time() < deadline - 10:
        for restart in range(100): # Increase restarts since it's faster now
            if time.time() > deadline:
                break
                
            colors = list(colors_29)
            for i in range(29):
                colors.append(random.randint(0, 1))
            
            current_conflicts = count_conflicts(30, colors)
            
            improved = True
            steps = 0
            # Temperature for simulated annealing or just hill climbing?
            # Hill climbing is safer.
            while improved and steps < 2000 and time.time() < deadline:
                improved = False
                steps += 1
                
                # Pick a random edge
                edge_idx = random.randint(0, len(colors) - 1)
                old_color = colors[edge_idx]
                new_color = 1 - old_color
                colors[edge_idx] = new_color
                
                new_conflicts = count_conflicts(30, colors)
                
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    improved = True
                    if current_conflicts == 0:
                        return {'n': 30, 'colors': colors}
                else:
                    colors[edge_idx] = old_color

    if not best_result:
        colors_17 = paley_colors(17)
        if colors_17 and is_valid(17, colors_17):
            best_result = {'n': 17, 'colors': colors_17}
        else:
            return {'n': 1, 'colors': []}

    return best_result
