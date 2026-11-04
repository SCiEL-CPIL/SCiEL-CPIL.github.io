import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 parts
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_terms():
        terms = []
        # M0
        terms.append({'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}})
        # M1
        terms.append({'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}})
        # M2
        terms.append({'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}})
        # M3
        terms.append({'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}})
        # M4
        terms.append({'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}})
        # M5
        terms.append({'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}})
        # M6
        terms.append({'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}})
        return terms

    terms = get_strassen_terms()
    
    col = 0
    for j_high in range(2):
        for l in range(4):
            for t_idx in range(7):
                term = terms[t_idx]
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (k_low, i), coeff in term['W'].items():
                    k = l * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
                
    # Greedy column removal
    # Check if removing column c keeps the decomposition valid
    # Tensor T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # We can precompute the target tensor T
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1 # Identity-like? No, multiplication tensor
                # T[i,j,k] is 1 if we multiply A[i,j] and B[j,k] to get C[i,k]
                # The tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k
                # So T[i,j,k] = 1 for all i,j,k?
                # Wait, the decomposition is for the multiplication map.
                # <T, A \otimes B> = <C, A \times B>
                # T_{i,j,k} = 1 for all i,j,k is correct for the standard basis.
                
    # Actually, let's just verify the current decomposition matches T
    # And try to remove columns
    
    def is_valid(U_list, V_list, W_list, active_cols):
        # Check if sum_{c in active} U_{ij,c} V_{jk,c} W_{ki,c} == 1 for all i,j,k
        # Note: W index is k*n + i.
        # U index is i*m + j.
        # V index is j*p + k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in active_cols:
                        s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                    if s != 1:
                        return False
        return True

    active = list(range(rank))
    # Try to remove columns
    changed = True
    while changed and time.time() < 140:
        changed = False
        for c in list(active):
            new_active = [x for x in active if x != c]
            if is_valid(U, V, W, new_active):
                active = new_active
                changed = True
                break # Restart loop to try removing another
                
    # Reconstruct matrices with only active columns
    if len(active) < rank:
        U_final = [[U[i][c] for c in active] for i in range(n*m)]
        V_final = [[V[i][c] for c in active] for i in range(m*p)]
        W_final = [[W[i][c] for c in active] for i in range(p*n)]
        return {"U": U_final, "V": V_final, "W": W_final}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
