import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    def make_paley_29():
        colors = []
        for i in range(29):
            for j in range(i + 1, 29):
                diff = (j - i) % 29
                colors.append(0 if diff in squares else 1)
        return colors

    base_colors_29 = make_paley_29()
    
    def list_to_matrix(colors, n):
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = colors[idx]
                mat[j][i] = colors[idx]
                idx += 1
        return mat

    def matrix_to_list(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    def find_violation(mat, n):
        adj0 = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i][j] == 0:
                    adj0[i].add(j)
                    adj0[j].add(i)
        
        for u in range(n):
            for v in range(u + 1, n):
                W = adj0[u].intersection(adj0[v])
                if len(W) < 4:
                    continue
                
                w_list = list(W)
                for idx_x in range(len(w_list)):
                    x = w_list[idx_x]
                    N_x = adj0[x].intersection(W)
                    if len(N_x) < 2:
                        continue
                    for y in N_x:
                        if y <= x:
                            continue
                        common = N_x.intersection(adj0[y]).intersection(W)
                        if len(common) >= 2:
                            c1, c2 = list(common)[:2]
                            return [u, v, x, y, c1, c2]
        return None

    n = 30
    mat_29 = list_to_matrix(base_colors_29, 29)
    
    # Try a few random initializations for the 30th vertex
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        mat_30 = [[0] * 30 for _ in range(30)]
        for i in range(29):
            for j in range(29):
                mat_30[i][j] = mat_29[i][j]
        
        for i in range(29):
            c = random.randint(0, 1)
            mat_30[i][29] = c
            mat_30[29][i] = c
        mat_30[29][29] = 0

        current_mat = mat_30
        
        # Local search
        # To avoid infinite loops, we can limit iterations per restart
        for iter_count in range(5000):
            if time.time() > deadline:
                break
                
            violation = find_violation(current_mat, n)
            if violation is None:
                return {'n': n, 'colors': matrix_to_list(current_mat, n)}
            
            u, v, x, y, c1, c2 = violation
            
            edges = [
                (u, x), (u, y), (v, x), (v, y),
                (u, c1), (u, c2), (v, c1), (v, c2),
                (x, c1), (x, c2), (y, c1), (y, c2)
            ]
            
            i, j = random.choice(edges)
            current_mat[i][j] = 1 - current_mat[i][j]
            current_mat[j][i] = current_mat[i][j]
            
            # If we are stuck, we might want to randomize more
            # But simple hill climbing often works for small n.

    # Fallback to 29
    return {'n': 29, 'colors': base_colors_29}

