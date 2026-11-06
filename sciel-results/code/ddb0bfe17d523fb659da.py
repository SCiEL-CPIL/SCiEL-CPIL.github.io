import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output serialization etc.

    def get_seed_solution():
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    # Helper to flatten a 2D matrix (upper triangle) to a list
    def mat_to_list(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    # Helper to verify a coloring (flat list) is valid
    def is_valid(n, colors):
        # Build matrix for easy access
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = colors[idx]
                idx += 1
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    def count_triangles(n, colors):
        """Count monochromatic triangles in O(n^3) time."""
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = colors[idx]
                idx += 1
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        count += 1
        return count

    def greedy_init(n):
        """Greedy vertex-by-vertex initialization."""
        mat = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                # Try each color and pick the one that creates the fewest mono triangles
                best_c = -1
                best_count = float('inf')
                for c in range(4):
                    mat[i][j] = mat[j][i] = c
                    # Count new mono triangles involving edge (i,j)
                    cnt = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if mat[i][k] == c and mat[j][k] == c:
                            cnt += 1
                    if cnt < best_count:
                        best_count = cnt
                        best_c = c
                mat[i][j] = mat[j][i] = best_c
        return mat_to_list(mat, n)

    def simulated_annealing(n, time_limit, init_method='greedy'):
        """Simulated annealing to find a valid 4-coloring."""
        if n < 2:
            return [0] * (n * (n - 1) // 2)
        
        def idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # Precompute all edge pairs for faster iteration
        edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        best_energy = float('inf')
        best_colors = None
        best_triangles = float('inf')
        
        for attempt in range(200):
            if time.time() > time_limit:
                break
            
            # Initialize
            if init_method == 'greedy':
                colors = greedy_init(n)
            else:
                colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Compute initial energy
            energy = count_triangles(n, colors)
            if energy == 0:
                return colors
            
            # Simulated annealing
            T = 50.0  # Initial temperature
            T_min = 0.01
            cooling = 0.9995
            
            for step in range(50000):
                if time.time() > time_limit:
                    break
                
                # Pick a random edge
                ei = random.randint(0, len(edge_pairs) - 1)
                i, j = edge_pairs[ei]
                eidx = idx(i, j)
                old_c = colors[eidx]
                
                # Try a new random color
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                
                # Compute delta energy
                # For each k, check if triangle (i,j,k) changes mono status
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ik_idx = idx(i, k)
                    jk_idx = idx(j, k)
                    c_ik = colors[ik_idx]
                    c_jk = colors[jk_idx]
                    
                    # Old mono status
                    old_mono = 1 if (c_ik == c_jk == old_c) else 0
                    # New mono status
                    new_mono = 1 if (c_ik == c_jk == new_c) else 0
                    
                    if old_mono != new_mono:
                        delta += (new_mono - old_mono)
                
                # Accept or reject
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    colors[eidx] = new_c
                    energy += delta
                
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min
                
                # Check for solution every 1000 steps
                if step % 1000 == 0:
                    if energy == 0:
                        return colors.copy()
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors.copy()
                    # Recompute energy periodically to avoid drift
                    if step % 5000 == 0:
                        energy = count_triangles(n, colors)
                        if energy == 0:
                            return colors.copy()
            
            # Final check
            if energy == 0:
                return colors.copy()
            
            # Try to improve best found so far
            if best_colors is not None and best_energy > 0:
                colors = best_colors
                energy = best_energy
                # Short local search
                for step in range(1000):
                    if time.time() > time_limit:
                        break
                    ei = random.randint(0, len(edge_pairs) - 1)
                    i, j = edge_pairs[ei]
                    eidx = idx(i, j)
                    old_c = colors[eidx]
                    
                    # Find best color for this edge
                    best_c = old_c
                    best_delta = 0
                    for new_c in range(4):
                        if new_c == old_c:
                            continue
                        delta = 0
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            ik_idx = idx(i, k)
                            jk_idx = idx(j, k)
                            c_ik = colors[ik_idx]
                            c_jk = colors[jk_idx]
                            old_mono = 1 if (c_ik == c_jk == old_c) else 0
                            new_mono = 1 if (c_ik == c_jk == new_c) else 0
                            if old_mono != new_mono:
                                delta += (new_mono - old_mono)
                        if delta < best_delta:
                            best_delta = delta
                            best_c = new_c
                    
                    if best_delta < 0:
                        colors[eidx] = best_c
                        energy += best_delta
                        if energy == 0:
                            return colors.copy()
        
        return None

    # Try to find coloring for n=50 down to 33
    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 3.0:
            break
        # Allocate time
        t_limit = time.time() + min(15.0, time_left * 0.85)
        res = simulated_annealing(n, t_limit, init_method='greedy')
        if res:
            # Verify
            if is_valid(n, res):
                return {'n': n, 'colors': res}
            else:
                print(f"Found invalid coloring for n={n}, ignoring.")
        
        # Also try random init if greedy didn't work
        time_left = deadline - time.time()
        if time_left < 3.0:
            break
        t_limit = time.time() + min(10.0, time_left * 0.85)
        res = simulated_annealing(n, t_limit, init_method='random')
        if res:
            if is_valid(n, res):
                return {'n': n, 'colors': res}
            else:
                print(f"Found invalid coloring for n={n} with random init, ignoring.")
            
    # Fallback to seed
    return get_seed_solution()

if __name__ == '__main__':
    res = construct()
    print(f"Result: n={res['n']}")
