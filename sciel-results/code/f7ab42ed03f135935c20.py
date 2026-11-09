import time
import random
import numpy as np

def construct():
    """
    Attempts to find a 4-coloring of K_50 without monochromatic triangles.
    Falls back to a known valid 4-coloring of K_32 if unsuccessful.
    """
    deadline = time.time() + 110.0
    
    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
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

    def extract_colors(C, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(C[i, j]))
        return colors

    def verify(C, n):
        for i in range(n):
            for j in range(i + 1, n):
                c = C[i, j]
                for k in range(j + 1, n):
                    if C[i, k] == c and C[j, k] == c:
                        return False
        return True

    n = 50
    C = np.zeros((n, n), dtype=np.int8)
    
    def init_random():
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                C[i, j] = c
                C[j, i] = c
                
    def calc_energy():
        E = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = C[i, j]
                # Count k > j such that C[i,k]==c and C[j,k]==c
                # Using numpy for speed
                # Mask k > j is not trivial with simple slicing without creating mask
                # Just loop k, n=50 is small
                for k in range(j + 1, n):
                    if C[i, k] == c and C[j, k] == c:
                        E += 1
        return E

    # Initialize
    init_random()
    E = calc_energy()
    best_E = E
    best_C = C.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    restarts = 0
    max_restarts = 50
    no_improve = 0
    improve_threshold = 20000
    
    while time.time() < deadline:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        c_old = int(C[u, v])
        c_new = random.randint(0, 3)
        if c_old == c_new:
            continue
            
        # Calculate Delta E
        # Count w != u, v such that C[u,w]==c_old and C[v,w]==c_old
        # And C[u,w]==c_new and C[v,w]==c_new
        
        # Vectorized
        row_u = C[u]
        row_v = C[v]
        
        count_old = np.sum((row_u == c_old) & (row_v == c_old))
        count_new = np.sum((row_u == c_new) & (row_v == c_new))
        
        # Adjust for self-loops (u and v)
        # C[u,u]=0, C[v,v]=0
        # If c_old == 0, w=u matches (0==0 and c_old==0), w=v matches (c_old==0 and 0==0)
        if c_old == 0:
            count_old -= 2
        if c_new == 0:
            count_new -= 2
            
        delta_E = count_new - count_old
        
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            C[u, v] = c_new
            C[v, u] = c_new
            E += delta_E
            
            if E < best_E:
                best_E = E
                best_C = C.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if E == 0:
                if verify(best_C, n):
                    return {'n': n, 'colors': extract_colors(best_C, n)}
        else:
            no_improve += 1

        T *= cooling
        if T < T_min:
            T = T_min
            
        if no_improve > improve_threshold:
            restarts += 1
            if restarts > max_restarts:
                break
            init_random()
            E = calc_energy()
            no_improve = 0
            T = 1.0

    # Fallback
    return get_fallback()
