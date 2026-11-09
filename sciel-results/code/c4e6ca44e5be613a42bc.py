"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 110
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    def get_color(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3
    
    # Build initial 32-vertex coloring
    n0 = 32
    colors0 = []
    for i in range(n0):
        for j in range(i + 1, n0):
            colors0.append(get_color(i, j))
    
    def build_matrix(colors, n):
        mat = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = colors[idx]
                idx += 1
        return mat
    
    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors
    
    mat0 = build_matrix(colors0, n0)
    
    def try_add_vertex(mat, n):
        """Try to add a new vertex. Returns assignment list of length n, or None."""
        assign = [-1] * n
        domains = [set([0,1,2,3]) for _ in range(n)]
        
        # Precompute degrees
        degrees = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if mat[i][j] >= 0:
                    degrees[i] += 1
                    degrees[j] += 1
        
        # Use static ordering by degree
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            i = order[pos]
            for c in [0,1,2,3]:
                if c not in domains[i]:
                    continue
                assign[i] = c
                saved = {}
                valid = True
                for k in range(pos+1, n):
                    j = order[k]
                    if mat[i][j] == c and c in domains[j]:
                        if j not in saved:
                            saved[j] = domains[j].copy()
                        domains[j].remove(c)
                        if not domains[j]:
                            valid = False
                            break
                
                if valid and backtrack(pos+1):
                    return True
                
                for j in saved:
                    domains[j] = saved[j]
                assign[i] = -1
            
            return False
        
        if backtrack(0):
            return assign
        return None
    
    best_n = n0
    best_mat = [row[:] for row in mat0]
    
    for target in range(33, 52):
        if time.time() > deadline - 5:
            break
        
        current_n = best_n
        current_mat = [row[:] for row in best_mat]
        
        added = 0
        for _ in range(target - current_n):
            if time.time() > deadline - 5:
                break
            assignment = try_add_vertex(current_mat, current_n)
            if assignment is None:
                break
            new_n = current_n + 1
            new_mat = [row[:] for row in current_mat]
            for i in range(current_n):
                new_mat[i].append(assignment[i])
            new_mat.append([-1] * current_n)
            for i in range(current_n):
                new_mat[current_n][i] = assignment[i]
            current_mat = new_mat
            current_n = new_n
            added += 1
        
        if added > 0:
            best_n = current_n
            best_mat = [row[:] for row in current_mat]
    
    best_colors = matrix_to_colors(best_mat, best_n)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
