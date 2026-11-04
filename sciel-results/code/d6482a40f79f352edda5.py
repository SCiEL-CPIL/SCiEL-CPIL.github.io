"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
start_time = time.time()
DEADLINE = 140.0

def construct():
    global start_time, DEADLINE
    # Check deadline
    if time.time() > DEADLINE:
        return {"U": [[0]], "V": [[0]], "W": [[0]]} # Fallback, though unlikely to be reached
        
    n, m, p = DIMENSIONS
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: Strassen 2x2x2 decomposition
    def strassen_2x2():
        terms = []
        # M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1} 
        terms.append((u, v, w))
        
        # M2
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {(1,0): 1, (1,1): -1} 
        terms.append((u, v, w))
        
        # M3
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {(0,1): 1, (1,1): 1} 
        terms.append((u, v, w))
        
        # M4
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {(0,0): 1, (1,0): 1} 
        terms.append((u, v, w))
        
        # M5
        u = {(0,0): 1, (0,1): 1}
        v = {(1,1): 1}
        w = {(0,0): -1, (0,1): 1} 
        terms.append((u, v, w))
        
        # M6
        u = {(1,0): 1, (0,0): -1}
        v = {(0,0): 1, (0,1): 1}
        w = {(1,1): 1} 
        terms.append((u, v, w))
        
        # M7
        u = {(0,1): 1, (1,1): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {(0,0): 1} 
        terms.append((u, v, w))
        
        return terms

    inner_terms = strassen_2x2()
    
    outer_terms = []
    outer_terms.append({
        'u': [(0,0,1), (1,1,1)],
        'v': [(0,0,1), (1,1,1)],
        'w': [(0,0,1), (1,1,1)]
    })
    outer_terms.append({
        'u': [(1,0,1), (1,1,1)],
        'v': [(0,0,1)],
        'w': [(1,0,1), (1,1,-1)]
    })
    outer_terms.append({
        'u': [(0,0,1)],
        'v': [(0,1,1), (1,1,-1)],
        'w': [(0,1,1), (1,1,1)]
    })
    outer_terms.append({
        'u': [(1,1,1)],
        'v': [(1,0,1), (0,0,-1)],
        'w': [(0,0,1), (1,0,1)]
    })
    outer_terms.append({
        'u': [(0,0,1), (0,1,1)],
        'v': [(1,1,1)],
        'w': [(0,0,-1), (0,1,1)]
    })
    outer_terms.append({
        'u': [(1,0,1), (0,0,-1)],
        'v': [(0,0,1), (0,1,1)],
        'w': [(1,1,1)]
    })
    outer_terms.append({
        'u': [(0,1,1), (1,1,-1)],
        'v': [(1,0,1), (1,1,1)],
        'w': [(0,0,1)]
    })
    
    for outer in outer_terms:
        for inner in inner_terms:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for br, bc, coeff_outer in outer['u']:
                for (li, lj), coeff_inner in inner[0].items():
                    gi = 2 * br + li
                    gj = 2 * bc + lj
                    key = (gi, gj)
                    u_dict[key] = u_dict.get(key, 0) + coeff_outer * coeff_inner
            
            for br, bc, coeff_outer in outer['v']:
                for (lj, lk), coeff_inner in inner[1].items():
                    gj = 2 * br + lj
                    gk = 2 * bc + lk
                    key = (gj, gk)
                    v_dict[key] = v_dict.get(key, 0) + coeff_outer * coeff_inner
            
            for br, bc, coeff_outer in outer['w']:
                for (lk, li), coeff_inner in inner[2].items():
                    gk = 2 * br + lk
                    gi = 2 * bc + li
                    key = (gk, gi)
                    w_dict[key] = w_dict.get(key, 0) + coeff_outer * coeff_inner
            
            for (i, j), val in u_dict.items():
                if val != 0:
                    U[i * m + j][col] = val
            for (j, k), val in v_dict.items():
                if val != 0:
                    V[j * p + k][col] = val
            for (k, i), val in w_dict.items():
                if val != 0:
                    W[k * n + i][col] = val
            
            col += 1
            
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
