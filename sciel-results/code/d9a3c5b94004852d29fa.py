"""Matrix multiplication 4x4x4 with rank-minimal decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin for safety
    
    n, m, p = 4, 4, 4
    
    def _verify_decomposition(U, V, W, verbose=False):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Condition: For all A, B, C = A*B iff C[i,k] = sum_r (U[:,r] . A_vec) * (V[:,r] . B_vec) * W[k*n+i, r]
        """
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank: return False
        for row in V:
            if len(row) != rank: return False
        for row in W:
            if len(row) != rank: return False
            
        # The identity is:
        # sum_r W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] = 1 if i'==i and k'==k else 0
        
        # We can verify this for all i,k,i',k',j'.
        # 16 * 16 * 4 = 1024 checks.
        # Each check is a dot product of length rank.
        
        # Let's use NumPy for speed.
        Uf = np.array(U, dtype=float)
        Vf = np.array(V, dtype=float)
        Wf = np.array(W, dtype=float)
        
        # Reshape for efficient computation?
        # U: (16, r), V: (16, r), W: (16, r)
        # We want to check T[i,k,i',k',j'] = sum_r W[k*4+i, r] U[i'*4+j', r] V[j'*4+k', r]
        # This is a 5D tensor.
        
        # For each r, we have vectors u_r, v_r, w_r.
        # The contribution to the tensor is w_r \otimes u_r \otimes v_r?
        # Specifically, the tensor T has indices (output_i, output_k, input_i, input_j, input_k).
        # T[oi, ok, ii, ij, ik] = sum_r W[ok*4+oi, r] U[ii*4+ij, r] V[ij*4+ik, r]
        # We want T[oi, ok, ii, ij, ik] = 1 if oi==ii and ok==ik, else 0.
        # Note that ij is the inner index, so it's free?
        # Wait, the matrix multiplication is C[i,k] = sum_j A[i,j] B[j,k].
        # So the tensor of the multiplication is T[i,k,j] = A[i,j]B[j,k]?
        # No, the tensor is defined such that <T, A \otimes B> = C.
        # C[i,k] = sum_{j} T[i,k,j] A[i,j] B[j,k].
        # So T[i,k,j] = 1 if the indices match?
        # Actually, the tensor T is a 3rd order tensor T_{ikj} where:
        # T_{ikj} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # In standard multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
        # So T_{ikj} = 1 for all j?
        # Let's check the decomposition formula:
        # C_{ik} = sum_r (sum_{i',j'} U_{i'j',r} A_{i'j'}) (sum_{j',k'} V_{j'k',r} B_{j'k'}) W_{ik,r}
        # = sum_{i',j',j',k'} (sum_r U_{i'j',r} V_{j'k',r} W_{ik,r}) A_{i'j'} B_{j'k'}
        # We want this to be sum_j A_{ij} B_{jk}.
        # So the coefficient of A_{i'j'} B_{j'k'} in C_{ik} must be:
        # 1 if i'==i, k'==k, and j' is the summation index?
        # No, the term is A_{i'j'} B_{j'k'}.
        # In the sum sum_j A_{ij} B_{jk}, the term is A_{i,j} B_{j,k}.
        # So for a fixed i,k, the non-zero terms are those with i'==i, k'==k, and j' is the index j.
        # So the coefficient is 1 if i'==i and k'==k (regardless of j'), and 0 otherwise.
        
        # So the condition is:
        # sum_r W_{ik,r} U_{i'j',r} V_{j'k',r} = 1 if i'==i and k'==k, else 0.
        # Note that j' is an index in U and V, but it is the same j' in both?
        # U is indexed by (i', j'), V by (j', k').
        # So the sum is over r. The indices i', j', k' are free variables on the LHS.
        # So for all i, k, i', j', k':
        # sum_r W[k*4+i, r] U[i'*4+j', r] V[j'*4+k', r] = 1 if i'==i and k'==k else 0.
        
        # Let's verify this with NumPy.
        # We can construct the tensor T from U, V, W.
        # T[i,k,i',j',k'] = sum_r W[k*4+i, r] U[i'*4+j', r] V[j'*4+k', r]
        
        # This is a bit slow in pure Python, but with NumPy we can do it.
        # However, 16*16*16*4*16 is too big?
        # 16*16*4 = 1024 checks.
        # Each check is a dot product of length rank (49).
        # 1024 * 49 ops is trivial.
        
        for i in range(n):
            for k in range(p):
                for i_p in range(n):
                    for k_p in range(p):
                        for j_p in range(m):
                            s = 0.0
                            for r in range(rank):
                                s += Wf[k*n+i][r] * Uf[i_p*m+j_p][r] * Vf[j_p*p+k_p][r]
                            expected = 1.0 if (i == i_p and k == k_p) else 0.0
                            if abs(s - expected) > 1e-9:
                                return False
        return True

    def _strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen (7 terms)."""
        U = []
        V = []
        W = []
        
        # A_vec = [A00, A01, A10, A11]
        # B_vec = [B00, B01, B10, B11]
        # C_vec indexing: W[k*2+i]
        
        # M1 = (A00 + A11)(B00 + B11)
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        # C00 += M1, C11 += M1
        W.append([1, 0, 0, 1]) # W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
        
        # M2 = (A10 + A11)B00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        # C10 += M2, C11 -= M2
        W.append([0, 1, 0, -1])
        
        # M3 = A00(B01 - B11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        # C01 += M3, C11 += M3
        W.append([0, 0, 1, 1])
        
        # M4 = A11(B10 - B00)
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        # C00 += M4, C10 += M4
        W.append([1, 1, 0, 0])
        
        # M5 = (A00 + A01)B11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        # C00 -= M5, C01 += M5
        W.append([-1, 0, 1, 0])
        
        # M6 = (A10 - A00)(B00 + B01)
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        # C11 += M6
        W.append([0, 0, 0, 1])
        
        # M7 = (A01 - A11)(B10 + B11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        # C00 += M7
        W.append([1, 0, 0, 0])
        
        return U, V, W

    def _build_4x4_strassen():
        """
        Build a rank-49 decomposition for 4x4x4.
        Strategy:
        Split 4x4 into 2x2 blocks of 2x2 matrices.
        A = [[A00, A01], [A10, A11]]
        B = [[B00, B01], [B10, B11]]
        C = A*B
        C00 = A00B00 + A01B10
        C01 = A00B01 + A01B11
        C10 = A10B00 + A11B10
        C11 = A10B01 + A11B11
        
        We use recursive Strassen.
        """
        
        U2, V2, W2 = _strassen_2x2()
        
        outer_terms = [
            # T1: (A00+A11)(B00+B11)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # T2: (A10+A11)B00
            ([(1,0,1), (1,1,1)], [(0,0,1)]),
            # T3: A00(B01-B11)
            ([(0,0,1)], [(0,1,1), (1,1,-1)]),
            # T4: A11(B10-B00)
            ([(1,1,1)], [(1,0,1), (0,0,-1)]),
            # T5: (A00+A01)B11
            ([(0,0,1), (0,1,1)], [(1,1,1)]),
            # T6: (A10-A00)(B00+B01)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            # T7: (A01-A11)(B10+B11)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ]
        
        recon = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)], # T1, T4, T5, T7
            (0,1): [(2,1), (4,1)],                # T3, T5
            (1,0): [(1,1), (3,1)],                # T2, T4
            (1,1): [(0,1), (1,-1), (2,1), (5,1)], # T1, T2, T3, T6
        }
        
        U4 = [[0.0]*49 for _ in range(16)]
        V4 = [[0.0]*49 for _ in range(16)]
        W4 = [[0.0]*49 for _ in range(16)]
        
        for t in range(7):
            L_blocks, R_blocks = outer_terms[t]
            for s in range(7):
                r_idx = t * 7 + s
                u_s = U2[s]
                v_s = V2[s]
                w_s = W2[s]
                
                for li in range(2):
                    for lj in range(2):
                        cu = u_s[li*2+lj]
                        if cu == 0: continue
                        for li_p in range(2):
                            for lj_p in range(2):
                                cv = v_s[li_p*2+lj_p]
                                if cv == 0: continue
                                
                                for br_L, bc_L, sgn_L in L_blocks:
                                    idx_A = (2*br_L+li)*4 + (2*bc_L+lj)
                                    coeff_A = sgn_L * cu
                                    
                                    for br_R, bc_R, sgn_R in R_blocks:
                                        idx_B = (2*br_R+li_p)*4 + (2*bc_R+lj_p)
                                        coeff_B = sgn_R * cv
                                        
                                        for ci in range(2):
                                            for cj in range(2):
                                                w_block = w_s[cj*2+ci]
                                                if w_block == 0: continue
                                                
                                                for (br, bc), pairs in recon.items():
                                                    for (tt, sign) in pairs:
                                                        if tt == t:
                                                            i_out = 2*br + ci
                                                            k_out = 2*bc + cj
                                                            w_idx = k_out * 4 + i_out
                                                            coeff_W = sign * w_block
                                                            
                                                            U4[idx_A][r_idx] += coeff_A
                                                            V4[idx_B][r_idx] += coeff_B
                                                            W4[w_idx][r_idx] += coeff_W
        
        # Convert to integers if possible
        U_int = [[int(round(x)) for x in row] for row in U4]
        V_int = [[int(round(x)) for x in row] for row in V4]
        W_int = [[int(round(x)) for x in row] for row in W4]
        
        if _verify_decomposition(U_int, V_int, W_int):
            return {"U": U_int, "V": V_int, "W": W_int}
        
        # Fallback to schoolbook
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    result = _build_4x4_strassen()
    return result
# EVOLVE-BLOCK-END
