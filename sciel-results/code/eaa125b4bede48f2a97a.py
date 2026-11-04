import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Global dimensions
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will collect the rank-1 components
    # Each component is (u_vec, v_vec, w_vec)
    # But it's easier to build the matrices directly by appending columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a component
    def add_component(u_vals, v_vals, w_vals):
        """
        u_vals: dict mapping local (n_loc, m_loc) to value
        v_vals: dict mapping local (m_loc, p_loc) to value
        w_vals: dict mapping local (p_loc, n_loc) to value
        """
        # We need to map local to global indices
        # This helper is called within the context of specific i, j, k blocks
        pass

    # We need to define the sub-decompositions
    
    # 1. Strassen 2x2x2
    # Returns list of (u, v, w) where u,v,w are dicts or lists of non-zero entries
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    def get_strassen_222():
        # Returns list of tuples (u_dict, v_dict, w_dict)
        # u_dict: {(n_loc, m_loc): val}
        # v_dict: {(m_loc, p_loc): val}
        # w_dict: {(p_loc, n_loc): val}
        
        comps = []
        
        # M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1}
        comps.append((u,v,w))
        
        # M2
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {(0,1): 1} # C21 is index 2? 
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W convention: W[k*n+i, r]. k is p index, i is n index.
        # Local W index: p_loc * 2 + n_loc.
        # C21 corresponds to n=1, p=0. Index 0*2 + 1 = 1? No.
        # C is 2x2. C[0,0], C[0,1], C[1,0], C[1,1].
        # W row index = p_loc * 2 + n_loc.
        # C[1,0] -> p=0, n=1 -> index 1.
        # C[1,1] -> p=1, n=1 -> index 3.
        # C[0,1] -> p=1, n=0 -> index 2.
        # C[0,0] -> p=0, n=0 -> index 0.
        
        # Let's re-verify Strassen output mapping.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My local C indices:
        # 0: (0,0) -> C11
        # 1: (0,1) -> C12 ? No, standard row-major is (0,0), (0,1), (1,0), (1,1).
        # But W convention is p*n + i.
        # p=0, n=0 -> 0. (C11)
        # p=0, n=1 -> 1. (C21)
        # p=1, n=0 -> 2. (C12)
        # p=1, n=1 -> 3. (C22)
        
        # So:
        # C11 is index 0.
        # C21 is index 1.
        # C12 is index 2.
        # C22 is index 3.
        
        # Let's re-evaluate Strassen terms with these indices.
        
        # M1: C11(+1), C22(+1). Indices 0, 3.
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {0: 1, 3: 1}
        comps.append((u,v,w))
        
        # M2: C21(+1). Index 1.
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {1: 1}
        comps.append((u,v,w))
        
        # M3: C12(+1). Index 2.
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {2: 1}
        comps.append((u,v,w))
        
        # M4: C11(+1), C21(+1). Indices 0, 1.
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {0: 1, 1: 1}
        comps.append((u,v,w))
        
        # M5: C11(-1), C12(+1). Indices 0, 2.
        u = {(0,0): 1, (0,1): 1}
        v = {(1,1): 1}
        w = {0: -1, 2: 1}
        comps.append((u,v,w))
        
        # M6: C22(+1). Index 3.
        u = {(1,0): 1, (0,0): -1}
        v = {(0,0): 1, (0,1): 1}
        w = {3: 1}
        comps.append((u,v,w))
        
        # M7: C11(+1), C22(+1). Indices 0, 3.
        u = {(0,1): 1, (1,1): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {0: 1, 3: 1}
        comps.append((u,v,w))
        
        return comps

    def get_schoolbook_221():
        # A 2x2, B 2x1, C 2x1
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # Local C indices:
        # p=0, n=0 -> 0 (C1)
        # p=0, n=1 -> 1 (C2)
        
        comps = []
        # Term 1: A11 B1 -> C1
        u = {(0,0): 1}
        v = {(0,0): 1}
        w = {0: 1}
        comps.append((u,v,w))
        
        # Term 2: A12 B2 -> C1
        u = {(0,1): 1}
        v = {(1,0): 1}
        w = {0: 1}
        comps.append((u,v,w))
        
        # Term 3: A21 B1 -> C2
        u = {(1,0): 1}
        v = {(0,0): 1}
        w = {1: 1}
        comps.append((u,v,w))
        
        # Term 4: A22 B2 -> C2
        u = {(1,1): 1}
        v = {(1,0): 1}
        w = {1: 1}
        comps.append((u,v,w))
        
        return comps

    def get_schoolbook_212():
        # A 2x1, B 1x2, C 2x2
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        # Local C indices:
        # p=0, n=0 -> 0 (C11)
        # p=0, n=1 -> 1 (C21)
        # p=1, n=0 -> 2 (C12)
        # p=1, n=1 -> 3 (C22)
        
        comps = []
        # A1 B1 -> C11
        u = {(0,0): 1}
        v = {(0,0): 1}
        w = {0: 1}
        comps.append((u,v,w))
        
        # A1 B2 -> C12
        u = {(0,0): 1}
        v = {(0,1): 1}
        w = {2: 1}
        comps.append((u,v,w))
        
        # A2 B1 -> C21
        u = {(1,0): 1}
        v = {(0,0): 1}
        w = {1: 1}
        comps.append((u,v,w))
        
        # A2 B2 -> C22
        u = {(1,0): 1}
        v = {(0,1): 1}
        w = {3: 1}
        comps.append((u,v,w))
        
        return comps

    def get_schoolbook_211():
        # A 2x1, B 1x1, C 2x1
        # C1 = A1 B
        # C2 = A2 B
        # Local C indices:
        # p=0, n=0 -> 0
        # p=0, n=1 -> 1
        
        comps = []
        u = {(0,0): 1}
        v = {(0,0): 1}
        w = {0: 1}
        comps.append((u,v,w))
        
        u = {(1,0): 1}
        v = {(0,0): 1}
        w = {1: 1}
        comps.append((u,v,w))
        
        return comps

    # Main loop
    # i: 0,1 (n block)
    # j: 0,1,2 (p block)
    # k: 0,1,2 (m block)
    
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                # Determine dimensions
                N = 2
                M = 2 if k < 2 else 1
                P = 2 if j < 2 else 1
                
                if M == 2 and P == 2:
                    comps = get_strassen_222()
                elif M == 2 and P == 1:
                    comps = get_schoolbook_221()
                elif M == 1 and P == 2:
                    comps = get_schoolbook_212()
                else: # M=1, P=1
                    comps = get_schoolbook_211()
                    
                for u_loc, v_loc, w_loc in comps:
                    # Create global vectors
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    
                    # Fill u_vec
                    for (n_loc, m_loc), val in u_loc.items():
                        n_glob = i*2 + n_loc
                        m_glob = m_offsets[k] + m_loc
                        idx = n_glob * m + m_glob
                        u_vec[idx] = val
                        
                    # Fill v_vec
                    for (m_loc, p_loc), val in v_loc.items():
                        m_glob = m_offsets[k] + m_loc
                        p_glob = p_offsets[j] + p_loc
                        idx = m_glob * p + p_glob
                        v_vec[idx] = val
                        
                    # Fill w_vec
                    for (p_loc, n_loc), val in w_loc.items():
                        p_glob = p_offsets[j] + p_loc
                        n_glob = i*2 + n_loc
                        idx = p_glob * n + n_glob
                        w_vec[idx] = val
                        
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                    
    rank = len(U_cols)
    # Transpose to get (n*m, rank)
    U = [list(row) for row in zip(*U_cols)]
    V = [list(row) for row in zip(*V_cols)]
    W = [list(row) for row in zip(*W_cols)]
    
    return {"U": U, "V": V, "W": W}

