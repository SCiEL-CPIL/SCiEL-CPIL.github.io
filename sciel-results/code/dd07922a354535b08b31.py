"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # 1. Build the known K32 4-coloring from K16 3-coloring (Clebsch)
    # This is a valid base.
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    _log, _x = {}, 1
    for _e in range(15):
        _log[_x] = _e
        _x = _mul(_x, 2)

    k16_mat = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(i + 1, 16):
            c = _log[i ^ j] % 3
            k16_mat[i][j] = k16_mat[j][i] = c

    # Double to K32
    k32_mat = [[0] * 32 for _ in range(32)]
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                k32_mat[i][j] = k32_mat[j][i] = k16_mat[i % 16][j % 16]
            else:
                k32_mat[i][j] = k32_mat[j][i] = 3

    best_n = 32
    best_mat = [row[:] for row in k32_mat]

    # 2. Incremental extension from 32 to 50
    def _count_mono_triangles(n, col_mat):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cij = col_mat[i][j]
                for k in range(j + 1, n):
                    if col_mat[i][k] == cij and col_mat[j][k] == cij:
                        count += 1
        return count

    def _delta_for_edge_change(n, col_mat, i, j, old_c, new_c):
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            cik = col_mat[i][k]
            cjk = col_mat[j][k]
            before = 1 if (old_c == cik == cjk) else 0
            after = 1 if (new_c == cik == cjk) else 0
            delta += after - before
        return delta

    def _sa_search(n, col_mat, timeout):
        """
        Simulated Annealing to fix mono triangles.
        Returns (success, final_mat)
        """
        start = time.time()
        cur_count = _count_mono_triangles(n, col_mat)
        if cur_count == 0:
            return True, col_mat

        T = 5.0
        T_min = 0.05
        cooling = 0.999
        
        best_col = [row[:] for row in col_mat]
        best_count = cur_count

        while time.time() - start < timeout and T > T_min:
            T *= cooling
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col_mat[i][j]
            new_c = (old_c + random.randint(1, 3)) % 4
            if new_c == old_c:
                continue
            
            delta = _delta_for_edge_change(n, col_mat, i, j, old_c, new_c)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                col_mat[i][j] = col_mat[j][i] = new_c
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_col = [row[:] for row in col_mat]
                    if best_count == 0:
                        return True, best_col
        
        return best_count == 0, best_col

    current_mat = [row[:] for row in k32_mat]
    current_n = 32

    for target_n in range(33, 51):
        if time.time() > deadline - 10.0:
            break
            
        # Extend current_mat to target_n
        # Initialize new edges greedily
        new_mat = [[0] * target_n for _ in range(target_n)]
        # Copy existing
        for i in range(current_n):
            for j in range(i + 1, current_n):
                new_mat[i][j] = new_mat[j][i] = current_mat[i][j]
        
        # Add new vertices one by one? Or all at once?
        # Since we are incrementing by 1, we add 1 vertex.
        # Wait, the loop goes 33, 34... so we add 1 vertex each step.
        v_new = current_n
        for u in range(current_n):
            # Greedy color assignment for (v_new, u)
            best_c = 0
            min_tri = float('inf')
            for c in range(4):
                tri = 0
                for w in range(u):
                    if new_mat[v_new][w] == c and new_mat[u][w] == c:
                        tri += 1
                # Also check w > u? No, we are filling row v_new.
                # We need to check triangles (v_new, u, w) for all w < v_new.
                # Since we are filling new_mat[v_new][u], we have new_mat[v_new][w] for w < u already set?
                # No, we are iterating u from 0 to current_n-1.
                # So for a fixed u, we have set new_mat[v_new][w] for w < u.
                # We need to check triangles with w < u and w > u (but w < current_n).
                # Triangles (v_new, u, w):
                # 1. w < u: new_mat[v_new][w] is set. new_mat[u][w] is set.
                # 2. w > u: new_mat[v_new][w] is NOT set yet (will be set later).
                # So greedy here is only partial.
                # Better: Just assign random or greedy based on immediate neighbors.
                pass
            
            # Simpler Greedy: Pick color that minimizes conflicts with already colored edges of v_new
            # Since we are filling v_new's row, we can just pick a color that is "safe" relative to u's neighbors.
            # Let's just pick a color that minimizes mono triangles with w < u.
            best_c = 0
            min_conflicts = 1000
            for c in range(4):
                conflicts = 0
                for w in range(u):
                    if new_mat[v_new][w] == c and new_mat[u][w] == c:
                        conflicts += 1
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_c = c
            new_mat[v_new][u] = new_mat[u][v_new] = best_c

        # Now run SA to fix any triangles created
        alloc_time = min(10.0, (deadline - time.time()) * 0.5)
        if alloc_time < 1.0:
            break
            
        success, fixed_mat = _sa_search(target_n, new_mat, alloc_time)
        
        if success:
            current_mat = fixed_mat
            current_n = target_n
            best_n = current_n
            best_mat = [row[:] for row in current_mat]
        else:
            # If we failed to fix triangles, we might still have a valid smaller n
            # But we can't use this invalid mat for the next step.
            # We should break or try a different strategy?
            # If we break, we return the last valid one.
            break

    # Convert best_mat to list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_mat[i][j])
            
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
