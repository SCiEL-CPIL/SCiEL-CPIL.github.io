import time
import random
import numpy as np
import math

def construct():
        start_time = time.time()
        deadline = start_time + 110

        def run_sa(N, time_limit):
            # N x N matrix
            mat = np.zeros((N, N), dtype=np.int8)
            for i in range(N):
                for j in range(i + 1, N):
                    c = random.randint(0, 3)
                    mat[i, j] = c
                    mat[j, i] = c
            
            # Count triangles
            # Vectorized count is hard to write quickly without bugs.
            # O(N^3) is 125k * 50 = 6.25M.
            # Let's do a simple loop for init, it's fast enough.
            count = 0
            for i in range(N):
                for j in range(i + 1, N):
                    c_ij = mat[i, j]
                    # Check k > j
                    for k in range(j + 1, N):
                        if c_ij == mat[j, k] == mat[k, i]:
                            count += 1

            T = 1.0
            T_min = 0.01
            cooling = 0.9995
            
            # Pre-compute indices? No, just loop.
            
            while time.time() < time_limit:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j: continue
                if i > j: i, j = j, i
                
                c_old = mat[i, j]
                c_new = random.randint(0, 3)
                while c_new == c_old:
                    c_new = random.randint(0, 3)
                
                # Delta calculation
                # Get rows i and j
                # Exclude i and j from consideration for k
                # We can just iterate k in Python, it's N=50, so 50 iterations.
                # NumPy overhead might be higher than 50 simple Python ops for small N.
                # Let's stick to Python loops for the inner delta, it's cleaner and likely fast enough for N=50.
                
                delta = 0
                # Optimize: store mat[i] and mat[j] in local vars?
                row_i = mat[i]
                row_j = mat[j]
                for k in range(N):
                    if k == i or k == j: continue
                    c_ik = row_i[k]
                    c_jk = row_j[k]
                    
                    was_mono = (c_old == c_ik == c_jk)
                    is_mono = (c_new == c_ik == c_jk)
                    
                    if is_mono:
                        delta += 1
                        if was_mono: delta -= 1 # No change
                    else:
                        if was_mono: delta -= 1
                        # else no change
                        
                # Simpler:
                # delta = (1 if is_mono else 0) - (1 if was_mono else 0)
                
                new_count = count + delta
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    mat[i, j] = c_new
                    mat[j, i] = c_new
                    count = new_count
                    
                    if count == 0:
                        break
                
                T *= cooling
                if T < T_min:
                    T = T_min

            # Convert to list
            colors = []
            for i in range(N):
                for j in range(i + 1, N):
                    colors.append(int(mat[i, j]))
            
            # Verify
            # Re-count to be sure
            v_count = 0
            for i in range(N):
                for j in range(i + 1, N):
                    c_ij = mat[i, j]
                    for k in range(j + 1, N):
                        if c_ij == mat[j, k] == mat[k, i]:
                            v_count += 1
                            
            if v_count == 0:
                return {'n': N, 'colors': colors}
            else:
                return None

        # Try N=50
        res = run_sa(50, deadline)
        if res:
            return res
            
        # If failed, try N=49
        if time.time() < deadline - 5:
            res = run_sa(49, deadline)
            if res:
                return res

        # If all else fails, return a dummy valid small graph?
        # K4 is easy.
        # But we want to maximize.
        # If we return invalid, score is 0.
        # If we return K4, score is 4.
        # Let's construct a valid K4.
        # 0-1:0, 0-2:0, 0-3:1, 1-2:1, 1-3:2, 2-3:3?
        # Triangles:
        # 012: 0,0,1 -> No
        # 013: 0,1,2 -> No
        # 023: 0,1,3 -> No
        # 123: 1,2,3 -> No
        # Colors: [0,0,1,1,2,3] for pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        return {'n': 4, 'colors': [0,0,1,1,2,3]}
