import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate base K32 coloring (Greenwood-Gleason doubled)
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
        
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
                
    best_n = 32
    best_colors = base_colors.copy()
    
    # Helper to get color of edge (i, j) from list
    # Index = i*n - (i+1)//2 + (j-i-1) ? No, simpler to use a 2D array or dict for search.
    # For n up to 50, a list of lists is fine.
    
    def list_to_matrix(colors, n):
        mat = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                mat[i][j] = colors[idx]
                mat[j][i] = colors[idx]
                idx += 1
        return mat

    def matrix_to_list(mat, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(mat[i][j])
        return colors

    # Check if matrix is valid (no mono triangles)
    def is_valid(mat, n):
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    # Count mono triangles for local search
    def count_mono_triangles(mat, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                # For each k > j, check if (i,k) and (j,k) are c
                # We can optimize by precomputing neighbors?
                # For n=50, O(n^3) is 125,000 ops, which is fast in Python if not too many iterations.
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        count += 1
        return count

    # Local search: Min-conflicts
    def local_search(mat, n, max_iter=1000):
        # Find all mono triangles
        # Strategy: Pick a random edge in a mono triangle and change its color
        # To be efficient, we can just iterate.
        
        current_cost = count_mono_triangles(mat, n)
        if current_cost == 0:
            return True, mat
            
        for _ in range(max_iter):
            if time.time() > deadline:
                return False, mat
                
            # Find a mono triangle
            # We can sample random triples to find conflicts faster?
            # Or just scan.
            # Scanning is deterministic and fast enough for n=50.
            found = False
            # To avoid bias, we can shuffle the order of edges?
            # Let's just pick a random edge and see if it's in a conflict.
            # Actually, finding ANY conflict:
            # We can maintain a list of conflicts? Too complex.
            # Just scan.
            for i in range(n):
                if time.time() > deadline:
                    return False, mat
                for j in range(i+1, n):
                    c = mat[i][j]
                    # Check if there is k such that (i,k) and (j,k) are c
                    # We can break early if found
                    for k in range(j+1, n):
                        if mat[i][k] == c and mat[j][k] == c:
                            # Found conflict (i,j,k) with color c
                            # Change color of (i,j) to something else?
                            # Or (i,k) or (j,k)?
                            # Randomly choose one of the 3 edges
                            edges = [(i,j), (i,k), (j,k)]
                            e1, e2 = random.choice(edges)
                            old_c = mat[e1][e2]
                            new_c = (old_c + 1) % 4
                            # Calculate delta cost? 
                            # Simplest: just change and recount? Expensive.
                            # Better: just change and hope it reduces.
                            # Min-conflicts usually requires delta evaluation.
                            # For n=50, recounting is ~0.1ms? 
                            # 1000 iters * 0.1ms = 100ms. Acceptable.
                            
                            mat[e1][e2] = new_c
                            mat[e2][e1] = new_c
                            new_cost = count_mono_triangles(mat, n)
                            if new_cost < current_cost:
                                current_cost = new_cost
                                if current_cost == 0:
                                    return True, mat
                            else:
                                # Revert
                                mat[e1][e2] = old_c
                                mat[e2][e1] = old_c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
                    
            if not found:
                # No conflict found? Should not happen if cost > 0
                break
                
        return current_cost == 0, mat

    # Extension strategy
    current_n = 32
    current_mat = list_to_matrix(best_colors, 32)
    
    for target_n in range(33, 51):
        if time.time() > deadline - 5.0:
            break
            
        success = False
        # Try multiple random initializations for the new vertex
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
                
            # Create new matrix
            new_mat = [[-1]*target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if i < current_n and j < current_n:
                        new_mat[i][j] = current_mat[i][j]
                        new_mat[j][i] = current_mat[i][j]
                    else:
                        # New edge involving vertex target_n-1
                        new_mat[i][j] = random.randint(0, 3)
                        new_mat[j][i] = new_mat[i][j]
                        
            # Local search on the new matrix
            valid, new_mat = local_search(new_mat, target_n, max_iter=500)
            if valid:
                current_n = target_n
                current_mat = new_mat
                best_n = target_n
                best_colors = matrix_to_list(new_mat, target_n)
                success = True
                break
                
        if not success:
            # If we failed to extend to target_n, we might still try next?
            # Usually if we can't extend K_n to K_{n+1}, it's hard.
            # But maybe a different random seed works.
            # We already tried 50 times.
            # We can break here if we want to be conservative, 
            # or continue to try higher n with a full restart?
            # Full restart for higher n is risky.
            # Let's break to save time, as extension is the main path.
            # However, sometimes a "jump" is possible with a different structure.
            # Given the time limit, breaking is safer.
            break

    return {'n': best_n, 'colors': best_colors}
