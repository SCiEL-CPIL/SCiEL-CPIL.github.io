    import numpy as np

    DIMENSIONS = [2, 2, 14]

    def construct():
        n, m, p = DIMENSIONS
        # n=2, m=2, p=14
        # Rank is 7 * (p/2) = 49
        
        num_blocks = p // 2
        rank = 7 * num_blocks
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen coefficients for 2x2x2
        # U_idx, V_idx, W_idx, coeff
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = [
            # (u_indices, v_indices, w_indices)
            # Each term is a list of (idx, coeff)
            [
                [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], # M1: C11+=1, C22+=1
            ],
            [
                [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)], # M2: C21+=1, C22-=1
            ],
            [
                [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)], # M3: C12+=1, C22+=1
            ],
            [
                [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)], # M4: C11+=1, C21+=1
            ],
            [
                [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)], # M5: C11-=1, C12+=1
            ],
            [
                [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], # M6: C22+=1
            ],
            [
                [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)], # M7: C11+=1
            ]
        ]
        
        # Note: My term definitions above need to be mapped to the correct W indices.
        # C indices:
        # C11 -> 0
        # C12 -> 1
        # C21 -> 2
        # C22 -> 3
        
        # Let's re-verify the W contributions for each M:
        # M1: C11 + 1, C22 + 1
        # M2: C21 + 1, C22 - 1
        # M3: C12 + 1, C22 + 1
        # M4: C11 + 1, C21 + 1
        # M5: C11 - 1, C12 + 1
        # M6: C22 + 1
        # M7: C11 + 1
        
        # Corrected terms structure: (U_mods, V_mods, W_mods)
        # U_mods: list of (idx, coeff)
        # V_mods: list of (idx, coeff)
        # W_mods: list of (idx, coeff)
        
        strassen_terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        ]
        
        for r in range(num_blocks):
            base_v = r * 2
            base_w = r * 4
            
            for t_idx in range(7):
                t = r * 7 + t_idx
                u_mods, v_mods, w_mods = strassen_terms[t_idx]
                
                for idx, coeff in u_mods:
                    U[idx, t] += coeff
                for idx, coeff in v_mods:
                    V[base_v + idx, t] += coeff
                for idx, coeff in w_mods:
                    W[base_w + idx, t] += coeff
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    