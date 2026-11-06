import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    sys.setrecursionlimit(10000)
    
    n_target = 50
    
    # Fallback 32-vertex construction
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = gf16_mul(x, 2)

    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors_32.append(log[i ^ j] % 3)
            else:
                colors_32.append(3)
                
    def try_solve(n):
        C = [[-1]*n for _ in range(n)]
        neighbors = [ [set() for _ in range(4)] for _ in range(n) ]
        
        # Symmetry breaking
        if n > 4:
            C[0][1] = 0; C[1][0] = 0
            neighbors[0][0].add(1); neighbors[1][0].add(0)
            C[0][2] = 1; C[2][0] = 1
            neighbors[0][1].add(2); neighbors[2][1].add(0)
            C[0][3] = 2; C[3][0] = 2
            neighbors[0][2].add(3); neighbors[3][2].add(0)
            C[0][4] = 3; C[4][0] = 3
            neighbors[0][3].add(4); neighbors[4][3].add(0)
            
        unassigned = []
        for i in range(n):
            for j in range(i+1, n):
                if C[i][j] == -1:
                    unassigned.append((i, j))
                    
        def get_valid_colors(i, j):
            valid = []
            for c in range(4):
                # Check if neighbors[i][c] and neighbors[j][c] intersect
                if not neighbors[i][c].intersection(neighbors[j][c]):
                    valid.append(c)
            return valid

        def backtrack(unassigned_list):
            if time.time() > deadline:
                return False
            if not unassigned_list:
                return True
                
            # MCV: Find variable with fewest valid colors
            min_count = 5
            min_idx = -1
            best_var = None
            best_colors = []
            
            # Sample a few to find MCV to save time?
            # Or just iterate all.
            # N=1225.
            for idx in range(len(unassigned_list)):
                i, j = unassigned_list[idx]
                colors = get_valid_colors(i, j)
                if len(colors) < min_count:
                    min_count = len(colors)
                    min_idx = idx
                    best_var = (i, j)
                    best_colors = colors
                    if min_count == 0:
                        break
                    if min_count == 1:
                        break
                        
            if min_count == 0:
                return False
                
            i, j = best_var
            unassigned_list.pop(min_idx)
            
            random.shuffle(best_colors)
            
            for c in best_colors:
                C[i][j] = c
                C[j][i] = c
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                
                if backtrack(unassigned_list):
                    return True
                
                C[i][j] = -1
                C[j][i] = -1
                neighbors[i][c].remove(j)
                neighbors[j][c].remove(i)
                
            unassigned_list.append((i, j))
            return False

        for attempt in range(20):
            if time.time() > deadline:
                break
            random.seed(attempt * 12345)
            # Reset
            for i in range(n):
                for j in range(n):
                    C[i][j] = -1
                for c in range(4):
                    neighbors[i][c].clear()
                    
            # Re-apply symmetry
            if n > 4:
                C[0][1] = 0; C[1][0] = 0
                neighbors[0][0].add(1); neighbors[1][0].add(0)
                C[0][2] = 1; C[2][0] = 1
                neighbors[0][1].add(2); neighbors[2][1].add(0)
                C[0][3] = 2; C[3][0] = 2
                neighbors[0][2].add(3); neighbors[3][2].add(0)
                C[0][4] = 3; C[4][0] = 3
                neighbors[0][3].add(4); neighbors[4][3].add(0)
                
            unassigned = []
            for i in range(n):
                for j in range(i+1, n):
                    if C[i][j] == -1:
                        unassigned.append((i, j))
                        
            if backtrack(unassigned):
                colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        colors.append(C[i][j])
                return colors
                
        return None

    # Try 50
    res = try_solve(50)
    if res:
        return {'n': 50, 'colors': res}
        
    # Try 49
    res = try_solve(49)
    if res:
        return {'n': 49, 'colors': res}
        
    # Fallback
    return {'n': 32, 'colors': colors_32}
