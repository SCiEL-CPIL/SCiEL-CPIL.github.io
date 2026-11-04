import numpy as np

def construct():
    n, m, p = 2, 3, 7
    
    # Initialize lists to store columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_start, j_start, k_start):
        """
        Adds a 2x2x2 Strassen block starting at indices i_start, j_start, k_start.
        Assumes i_start, j_start, k_start are 0-based and the block fits within bounds.
        """
        # Strassen coefficients for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A part), V (B part), W (C part)
        # Local indices 0,1 map to global i_start+0, i_start+1 etc.
        
        # We define the 7 terms as lists of (local_i, local_j, coeff) for U
        # (local_j, local_k, coeff) for V
        # (local_k, local_i, coeff) for W
        
        terms = []
        
        # Term 1
        terms.append({
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00 += 1, C11 += 1
        })
        # Term 2
        terms.append({
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 += 1, C11 -= 1
        })
        # Term 3
        terms.append({
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)] # C01 += 1, C11 += 1
        })
        # Term 4
        terms.append({
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 += 1, C10 += 1
        })
        # Term 5
        terms.append({
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 -= 1, C01 += 1
        })
        # Term 6
        terms.append({
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 += 1
        })
        # Term 7
        terms.append({
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00 += 1, C11 += 1
        })
        
        for t in terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for li, lj, coeff in t['U']:
                gi = i_start + li
                gj = j_start + lj
                u_vec[gi * m + gj] += coeff
            
            for lj, lk, coeff in t['V']:
                gj = j_start + lj
                gk = k_start + lk
                v_vec[gj * p + gk] += coeff
            
            for lk, li, coeff in t['W']:
                gk = k_start + lk
                gi = i_start + li
                w_vec[gk * n + gi] += coeff
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """
        Adds schoolbook terms for i in [i_start, i_end), j in [j_start, j_end), k in [k_start, k_end).
        """
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (Rank 7)
    add_strassen_block(0, 0, 0)
    
    # Block A: j=2, i in {0,1}, k in {0..6} -> Schoolbook (Rank 14)
    add_schoolbook_block(0, 2, 2, 3, 0, 7)
    
    # Block B1: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen (Rank 7)
    add_strassen_block(0, 0, 2)
    
    # Block B2: i in {0,1}, j in {0,1}, k in {4,5,6} -> Schoolbook (Rank 12)
    add_schoolbook_block(0, 2, 0, 2, 4, 7)
    
    # Convert lists to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
