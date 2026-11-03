"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Laderman algorithm for 3x3 matrix multiplication (rank 23)
    def laderman_3x3():
        """
        Laderman's algorithm for 3x3 matrix multiplication.
        Uses 23 multiplications instead of 27.
        
        The 23 bilinear forms are:
        M1 = A00 * B00
        M2 = A01 * B01
        M3 = A02 * B02
        M4 = A10 * B10
        M5 = A11 * B11
        M6 = A12 * B12
        M7 = A20 * B20
        M8 = A21 * B21
        M9 = A22 * B22
        M10 = (A00 + A01) * (B10 - B00)
        M11 = (A00 + A02) * (B20 - B00)
        M12 = (A10 + A11) * (B01 - B10)
        M13 = (A10 + A12) * (B02 - B10)
        M14 = (A20 + A21) * (B01 - B20)
        M15 = (A20 + A22) * (B02 - B20)
        M16 = (A11 + A22) * (B11 - B22)
        M17 = (A11 + A12 + A21 + A22) * (B11 - B12 - B21 + B22)
        M18 = (A00 + A11 + A22) * (B12 - B02)
        M19 = (A00 + A11 + A22) * (B21 - B01)
        M20 = (A00 + A11 + A22) * (B12 - B21)
        M21 = (A01 + A12 + A20) * (B12 - B20)
        M22 = (A02 + A10 + A21) * (B10 - B02)
        M23 = (A02 + A10 + A21) * (B20 - B12)
        
        Wait, let me use the standard Laderman formulation.
        """
        # Laderman's 23-term decomposition for 3x3 matrices
        # A is 3x3, B is 3x3, C = A * B is 3x3
        
        # The 23 multiplication terms
        terms = []
        
        # Direct terms (9)
        # M1: A[0,0] * B[0,0]
        terms.append({
            'a': {0: [(0,0)]},
            'b': {0: [(0,0)]},
            'c': {0: [(0,0)]}
        })
        # M2: A[0,1] * B[1,1]
        terms.append({
            'a': {0: [(0,1)]},
            'b': {0: [(1,1)]},
            'c': {0: [(0,1)]}
        })
        # M3: A[0,2] * B[2,2]
        terms.append({
            'a': {0: [(0,2)]},
            'b': {0: [(2,2)]},
            'c': {0: [(0,2)]}
        })
        # M4: A[1,0] * B[0,0]
        terms.append({
            'a': {0: [(1,0)]},
            'b': {0: [(0,0)]},
            'c': {0: [(1,0)]}
        })
        # M5: A[1,1] * B[1,1]
        terms.append({
            'a': {0: [(1,1)]},
            'b': {0: [(1,1)]},
            'c': {0: [(1,1)]}
        })
        # M6: A[1,2] * B[2,2]
        terms.append({
            'a': {0: [(1,2)]},
            'b': {0: [(2,2)]},
            'c': {0: [(1,2)]}
        })
        # M7: A[2,0] * B[0,0]
        terms.append({
            'a': {0: [(2,0)]},
            'b': {0: [(0,0)]},
            'c': {0: [(2,0)]}
        })
        # M8: A[2,1] * B[1,1]
        terms.append({
            'a': {0: [(2,1)]},
            'b': {0: [(1,1)]},
            'c': {0: [(2,1)]}
        })
        # M9: A[2,2] * B[2,2]
        terms.append({
            'a': {0: [(2,2)]},
            'b': {0: [(2,2)]},
            'c': {0: [(2,2)]}
        })
        
        # Cross terms (14)
        # M10: (A[0,0] + A[0,1]) * (B[1,0] - B[0,0])
        terms.append({
            'a': {1: [(0,0)], 2: [(0,1)]},
            'b': {1: [(1,0)], 2: [(0,0), -1]},
            'c': {1: [(0,1)]}
        })
        # M11: (A[0,0] + A[0,2]) * (B[2,0] - B[0,0])
        terms.append({
            'a': {1: [(0,0)], 2: [(0,2)]},
            'b': {1: [(2,0)], 2: [(0,0), -1]},
            'c': {1: [(0,2)]}
        })
        # M12: (A[1,0] + A[1,1]) * (B[0,1] - B[1,0])
        terms.append({
            'a': {1: [(1,0)], 2: [(1,1)]},
            'b': {1: [(0,1)], 2: [(1,0), -1]},
            'c': {1: [(1,1)]}
        })
        # M13: (A[1,0] + A[1,2]) * (B[0,2] - B[1,0])
        terms.append({
            'a': {1: [(1,0)], 2: [(1,2)]},
            'b': {1: [(0,2)], 2: [(1,0), -1]},
            'c': {1: [(1,2)]}
        })
        # M14: (A[2,0] + A[2,1]) * (B[0,1] - B[2,0])
        terms.append({
            'a': {1: [(2,0)], 2: [(2,1)]},
            'b': {1: [(0,1)], 2: [(2,0), -1]},
            'c': {1: [(2,1)]}
        })
        # M15: (A[2,0] + A[2,2]) * (B[0,2] - B[2,0])
        terms.append({
            'a': {1: [(2,0)], 2: [(2,2)]},
            'b': {1: [(0,2)], 2: [(2,0), -1]},
            'c': {1: [(2,2)]}
        })
        # M16: (A[1,1] + A[2,2]) * (B[1,1] - B[2,2])
        terms.append({
            'a': {1: [(1,1)], 2: [(2,2)]},
            'b': {1: [(1,1)], 2: [(2,2), -1]},
            'c': {1: [(1,2)], 2: [(2,1)], 3: [(1,1), -1], 4: [(2,2), -1]}
        })
        # M17: (A[1,1] + A[1,2] + A[2,1] + A[2,2]) * (B[1,1] - B[1,2] - B[2,1] + B[2,2])
        terms.append({
            'a': {1: [(1,1)], 2: [(1,2)], 3: [(2,1)], 4: [(2,2)]},
            'b': {1: [(1,1)], 2: [(1,2), -1], 3: [(2,1), -1], 4: [(2,2)]},
            'c': {1: [(1,2), 1], 2: [(2,1), 1], 3: [(1,1), -1], 4: [(2,2), -1]}
        })
        # M18: (A[0,0] + A[1,1] + A[2,2]) * (B[1,2] - B[0,2])
        terms.append({
            'a': {1: [(0,0)], 2: [(1,1)], 3: [(2,2)]},
            'b': {1: [(1,2)], 2: [(0,2), -1]},
            'c': {1: [(0,2)]}
        })
        # M19: (A[0,0] + A[1,1] + A[2,2]) * (B[2,1] - B[0,1])
        terms.append({
            'a': {1: [(0,0)], 2: [(1,1)], 3: [(2,2)]},
            'b': {1: [(2,1)], 2: [(0,1), -1]},
            'c': {1: [(0,1)]}
        })
        # M20: (A[0,0] + A[1,1] + A[2,2]) * (B[1,2] - B[2,1])
        terms.append({
            'a': {1: [(0,0)], 2: [(1,1)], 3: [(2,2)]},
            'b': {1: [(1,2)], 2: [(2,1), -1]},
            'c': {1: [(1,2)], 2: [(2,1), -1]}
        })
        # M21: (A[0,1] + A[1,2] + A[2,0]) * (B[1,2] - B[2,0])
        terms.append({
            'a': {1: [(0,1)], 2: [(1,2)], 3: [(2,0)]},
            'b': {1: [(1,2)], 2: [(2,0), -1]},
            'c': {1: [(0,2)]}
        })
        # M22: (A[0,2] + A[1,0] + A[2,1]) * (B[1,0] - B[0,2])
        terms.append({
            'a': {1: [(0,2)], 2: [(1,0)], 3: [(2,1)]},
            'b': {1: [(1,0)], 2: [(0,2), -1]},
            'c': {1: [(2,0)]}
        })
        # M23: (A[0,2] + A[1,0] + A[2,1]) * (B[2,0] - B[1,2])
        terms.append({
            'a': {1: [(0,2)], 2: [(1,0)], 3: [(2,1)]},
            'b': {1: [(2,0)], 2: [(1,2), -1]},
            'c': {1: [(0,0)]}
        })
        
        return terms
    
    # Build U, V, W matrices from a set of terms
    def build_from_terms(terms):
        U_list = []
        V_list = []
        W_list = []
        
        for term in terms:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # Parse term structure
            # term['a']: {coeff_index: [(i,j), coeff]} or {coeff_index: [(i,j), coeff1, coeff2]}
            # This is getting complicated. Let me use a simpler representation.
            pass
        
        return np.array(U_list).T, np.array(V_list).T, np.array(W_list).T, len(terms)
    
    # Simpler approach: Define Laderman terms explicitly with vector contributions
    def laderman_terms():
        """
        Returns list of (u_vec, v_vec, w_vec) for Laderman 3x3 algorithm.
        u_vec has length 9 (3x3 A matrix indices)
        v_vec has length 9 (3x3 B matrix indices)
        w_vec has length 9 (3x3 C matrix indices)
        """
        terms = []
        
        # Helper to create index for i,j in 3x3
        def idx(i, j):
            return i * 3 + j
        
        # M1: A[0,0] * B[0,0] -> C[0,0]
        u1 = np.zeros(9); v1 = np.zeros(9); w1 = np.zeros(9)
        u1[idx(0,0)] = 1; v1[idx(0,0)] = 1; w1[idx(0,0)] = 1
        terms.append((u1, v1, w1))
        
        # M2: A[0,1] * B[1,1] -> C[0,1]
        u2 = np.zeros(9); v2 = np.zeros(9); w2 = np.zeros(9)
        u2[idx(0,1)] = 1; v2[idx(1,1)] = 1; w2[idx(0,1)] = 1
        terms.append((u2, v2, w2))
        
        # M3: A[0,2] * B[2,2] -> C[0,2]
        u3 = np.zeros(9); v3 = np.zeros(9); w3 = np.zeros(9)
        u3[idx(0,2)] = 1; v3[idx(2,2)] = 1; w3[idx(0,2)] = 1
        terms.append((u3, v3, w3))
        
        # M4: A[1,0] * B[0,0] -> C[1,0]
        u4 = np.zeros(9); v4 = np.zeros(9); w4 = np.zeros(9)
        u4[idx(1,0)] = 1; v4[idx(0,0)] = 1; w4[idx(1,0)] = 1
        terms.append((u4, v4, w4))
        
        # M5: A[1,1] * B[1,1] -> C[1,1]
        u5 = np.zeros(9); v5 = np.zeros(9); w5 = np.zeros(9)
        u5[idx(1,1)] = 1; v5[idx(1,1)] = 1; w5[idx(1,1)] = 1
        terms.append((u5, v5, w5))
        
        # M6: A[1,2] * B[2,2] -> C[1,2]
        u6 = np.zeros(9); v6 = np.zeros(9); w6 = np.zeros(9)
        u6[idx(1,2)] = 1; v6[idx(2,2)] = 1; w6[idx(1,2)] = 1
        terms.append((u6, v6, w6))
        
        # M7: A[2,0] * B[0,0] -> C[2,0]
        u7 = np.zeros(9); v7 = np.zeros(9); w7 = np.zeros(9)
        u7[idx(2,0)] = 1; v7[idx(0,0)] = 1; w7[idx(2,0)] = 1
        terms.append((u7, v7, w7))
        
        # M8: A[2,1] * B[1,1] -> C[2,1]
        u8 = np.zeros(9); v8 = np.zeros(9); w8 = np.zeros(9)
        u8[idx(2,1)] = 1; v8[idx(1,1)] = 1; w8[idx(2,1)] = 1
        terms.append((u8, v8, w8))
        
        # M9: A[2,2] * B[2,2] -> C[2,2]
        u9 = np.zeros(9); v9 = np.zeros(9); w9 = np.zeros(9)
        u9[idx(2,2)] = 1; v9[idx(2,2)] = 1; w9[idx(2,2)] = 1
        terms.append((u9, v9, w9))
        
        # M10: (A[0,0]+A[0,1]) * (B[1,0]-B[0,0]) -> C[0,1]
        u10 = np.zeros(9); v10 = np.zeros(9); w10 = np.zeros(9)
        u10[idx(0,0)] = 1; u10[idx(0,1)] = 1
        v10[idx(1,0)] = 1; v10[idx(0,0)] = -1
        w10[idx(0,1)] = 1
        terms.append((u10, v10, w10))
        
        # M11: (A[0,0]+A[0,2]) * (B[2,0]-B[0,0]) -> C[0,2]
        u11 = np.zeros(9); v11 = np.zeros(9); w11 = np.zeros(9)
        u11[idx(0,0)] = 1; u11[idx(0,2)] = 1
        v11[idx(2,0)] = 1; v11[idx(0,0)] = -1
        w11[idx(0,2)] = 1
        terms.append((u11, v11, w11))
        
        # M12: (A[1,0]+A[1,1]) * (B[0,1]-B[1,0]) -> C[1,1]
        u12 = np.zeros(9); v12 = np.zeros(9); w12 = np.zeros(9)
        u12[idx(1,0)] = 1; u12[idx(1,1)] = 1
        v12[idx(0,1)] = 1; v12[idx(1,0)] = -1
        w12[idx(1,1)] = 1
        terms.append((u12, v12, w12))
        
        # M13: (A[1,0]+A[1,2]) * (B[0,2]-B[1,0]) -> C[1,2]
        u13 = np.zeros(9); v13 = np.zeros(9); w13 = np.zeros(9)
        u13[idx(1,0)] = 1; u13[idx(1,2)] = 1
        v13[idx(0,2)] = 1; v13[idx(1,0)] = -1
        w13[idx(1,2)] = 1
        terms.append((u13, v13, w13))
        
        # M14: (A[2,0]+A[2,1]) * (B[0,1]-B[2,0]) -> C[2,1]
        u14 = np.zeros(9); v14 = np.zeros(9); w14 = np.zeros(9)
        u14[idx(2,0)] = 1; u14[idx(2,1)] = 1
        v14[idx(0,1)] = 1; v14[idx(2,0)] = -1
        w14[idx(2,1)] = 1
        terms.append((u14, v14, w14))
        
        # M15: (A[2,0]+A[2,2]) * (B[0,2]-B[2,0]) -> C[2,2]
        u15 = np.zeros(9); v15 = np.zeros(9); w15 = np.zeros(9)
        u15[idx(2,0)] = 1; u15[idx(2,2)] = 1
        v15[idx(0,2)] = 1; v15[idx(2,0)] = -1
        w15[idx(2,2)] = 1
        terms.append((u15, v15, w15))
        
        # M16: (A[1,1]+A[2,2]) * (B[1,1]-B[2,2]) -> C[1,2] - C[2,1]
        u16 = np.zeros(9); v16 = np.zeros(9); w16 = np.zeros(9)
        u16[idx(1,1)] = 1; u16[idx(2,2)] = 1
        v16[idx(1,1)] = 1; v16[idx(2,2)] = -1
        w16[idx(1,2)] = 1; w16[idx(2,1)] = -1
        terms.append((u16, v16, w16))
        
        # M17: (A[1,1]+A[1,2]+A[2,1]+A[2,2]) * (B[1,1]-B[1,2]-B[2,1]+B[2,2])
        # -> C[1,2] + C[2,1] - C[1,1] - C[2,2]
        u17 = np.zeros(9); v17 = np.zeros(9); w17 = np.zeros(9)
        u17[idx(1,1)] = 1; u17[idx(1,2)] = 1; u17[idx(2,1)] = 1; u17[idx(2,2)] = 1
        v17[idx(1,1)] = 1; v17[idx(1,2)] = -1; v17[idx(2,1)] = -1; v17[idx(2,2)] = 1
        w17[idx(1,2)] = 1; w17[idx(2,1)] = 1; w17[idx(1,1)] = -1; w17[idx(2,2)] = -1
        terms.append((u17, v17, w17))
        
        # M18: (A[0,0]+A[1,1]+A[2,2]) * (B[1,2]-B[0,2]) -> C[0,2]
        u18 = np.zeros(9); v18 = np.zeros(9); w18 = np.zeros(9)
        u18[idx(0,0)] = 1; u18[idx(1,1)] = 1; u18[idx(2,2)] = 1
        v18[idx(1,2)] = 1; v18[idx(0,2)] = -1
        w18[idx(0,2)] = 1
        terms.append((u18, v18, w18))
        
        # M19: (A[0,0]+A[1,1]+A[2,2]) * (B[2,1]-B[0,1]) -> C[0,1]
        u19 = np.zeros(9); v19 = np.zeros(9); w19 = np.zeros(9)
        u19[idx(0,0)] = 1; u19[idx(1,1)] = 1; u19[idx(2,2)] = 1
        v19[idx(2,1)] = 1; v19[idx(0,1)] = -1
        w19[idx(0,1)] = 1
        terms.append((u19, v19, w19))
        
        # M20: (A[0,0]+A[1,1]+A[2,2]) * (B[1,2]-B[2,1]) -> C[1,2] - C[2,1]
        u20 = np.zeros(9); v20 = np.zeros(9); w20 = np.zeros(9)
        u20[idx(0,0)] = 1; u20[idx(1,1)] = 1; u20[idx(2,2)] = 1
        v20[idx(1,2)] = 1; v20[idx(2,1)] = -1
        w20[idx(1,2)] = 1; w20[idx(2,1)] = -1
        terms.append((u20, v20, w20))
        
        # M21: (A[0,1]+A[1,2]+A[2,0]) * (B[1,2]-B[2,0]) -> C[0,2]
        u21 = np.zeros(9); v21 = np.zeros(9); w21 = np.zeros(9)
        u21[idx(0,1)] = 1; u21[idx(1,2)] = 1; u21[idx(2,0)] = 1
        v21[idx(1,2)] = 1; v21[idx(2,0)] = -1
        w21[idx(0,2)] = 1
        terms.append((u21, v21, w21))
        
        # M22: (A[0,2]+A[1,0]+A[2,1]) * (B[1,0]-B[0,2]) -> C[2,0]
        u22 = np.zeros(9); v22 = np.zeros(9); w22 = np.zeros(9)
        u22[idx(0,2)] = 1; u22[idx(1,0)] = 1; u22[idx(2,1)] = 1
        v22[idx(1,0)] = 1; v22[idx(0,2)] = -1
        w22[idx(2,0)] = 1
        terms.append((u22, v22, w22))
        
        # M23: (A[0,2]+A[1,0]+A[2,1]) * (B[2,0]-B[1,2]) -> C[0,0]
        u23 = np.zeros(9); v23 = np.zeros(9); w23 = np.zeros(9)
        u23[idx(0,2)] = 1; u23[idx(1,0)] = 1; u23[idx(2,1)] = 1
        v23[idx(2,0)] = 1; v23[idx(1,2)] = -1
        w23[idx(0,0)] = 1
        terms.append((u23, v23, w23))
        
        return terms
    
    # Verify Laderman decomposition
    def verify_laderman(terms):
        # Test that sum of u_i * v_i * w_i = identity for 3x3 matrix mult
        # C[i,j] = sum_k A[i,k] * B[k,j]
        for i in range(3):
            for j in range(3):
                lhs = 1.0 if i == j else 0.0  # Wait, this is wrong
                # Actually for matrix multiplication, C[i,j] = sum_k A[i,k] * B[k,j]
                # The tensor is T[i,j,k] = delta[i,k] * delta[k,j] = delta[i,j] if k matches
                # Actually the bilinear form is: C[i,j] = sum_k A[i,k] B[k,j]
                # So the tensor has T[i,k,j] = 1 for all i,k,j (it's the all-ones tensor for the product)
                # Wait no. For matrix multiplication, C = A * B means C[i,j] = sum_k A[i,k] B[k,j]
                # The structure tensor has T[i,k,j] = 1 for all i,k,j? No.
                # Actually, the tensor for matrix multiplication M_{p,q,r} has T[i,j,k] = delta[i,j] if we view it as
                # C[i,k] = sum_j A[i,j] B[j,k]
                # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
                
                # Let me think again. We want to verify that for random A, B:
                # sum_t (sum_ij U[i,j,t] * A[i,j]) * (sum_kl V[k,l,t] * B[k,l]) * W[i',k',t] 
                # gives C[i',k'] = sum_j A[i',j] B[j,k']
                
                # The condition is: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for all i,j,k
                # (using the convention where U is indexed by (i,j), V by (j,k), W by (k,i))
                
                lhs = 1.0
                rhs = 0.0
                for u, v, w in terms:
                    rhs += u[i*3+j] * v[j*3+k] * w[k*3+i]
                if abs(lhs - rhs) > 1e-9:
                    return False, (i, j, k, lhs, rhs)
        return True, None
    
    # Build the 3x5x7 decomposition using Laderman for a 3x3 block
    # The 3x5 matrix A and 5x7 matrix B
    # We can partition: A is 3x5, B is 5x7
    # Think of A as 3x(5) and B as (5)x7
    # We can use Laderman on a 3x3 sub-block where A contributes 3x3 and B contributes 3x3
    # But B is 5x7, so we need to be careful.
    
    # Alternative: Use the fact that 3x5 x 5x7 can be decomposed by:
    # - Using Laderman for the "core" 3x3 multiplication
    # - Plus additional terms for the extra dimensions
    
    # Let's think of it as:
    # A is 3x5, so A[i,j] for i=0,1,2 and j=0,1,2,3,4
    # B is 5x7, so B[j,k] for j=0,1,2,3,4 and k=0,1,2,3,4,5,6
    # C is 3x7, so C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can split j into two groups: j in {0,1,2} and j in {3,4}
    # For j in {0,1,2}: This is a 3x3 (A part) times 3x7 (B part)
    # For j in {3,4}: This is a 3x2 (A part) times 2x7 (B part)
    
    # Actually, let me use a different approach.
    # Use Strassen-style decomposition on blocks.
    
    # The 3x5 x 5x7 multiplication can be seen as:
    # [A0 A1 A2] * [B0; B1; B2] where A0 is 3x2, A1 is 3x1, A2 is 3x2
    # and B0 is 2x7, B1 is 1x7, B2 is 2x7
    
    # This is getting complicated. Let me use a simpler strategy:
    # Start with schoolbook (rank 105) and try to find better decompositions.
    
    # For now, let's use the Strassen-based approach from the current program
    # but also try Laderman for a 3x3 subproblem.
    
    # Define the 6 Strassen blocks from the current program
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    U = []
    V = []
    W = []
    
    def add_strassen_block(I, J, K):
        # Map local indices 0,1 to global indices
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # Strassen 7 terms for 2x2x2
        terms = []
        
        # M1
        u1 = np.zeros(n*m)
        v1 = np.zeros(m*p)
        w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0
        u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0
        v1[j1*p + k1] = 1.0
        w1[k0*n + i0] = 1.0
        w1[k1*n + i1] = 1.0
        terms.append((u1, v1, w1))
        
        # M2
        u2 = np.zeros(n*m)
        v2 = np.zeros(m*p)
        w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1.0
        u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + i1] = 1.0
        w2[k1*n + i1] = -1.0
        terms.append((u2, v2, w2))
        
        # M3
        u3 = np.zeros(n*m)
        v3 = np.zeros(m*p)
        w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0
        v3[j1*p + k1] = -1.0
        w3[k1*n + i0] = 1.0
        w3[k1*n + i1] = 1.0
        terms.append((u3, v3, w3))
        
        # M4
        u4 = np.zeros(n*m)
        v4 = np.zeros(m*p)
        w4 = np.zeros(p*n)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0
        v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0
        w4[k0*n + i1] = 1.0
        terms.append((u4, v4, w4))
        
        # M5
        u5 = np.zeros(n*m)
        v5 = np.zeros(m*p)
        w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1.0
        u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        w5[k0*n + i0] = -1.0
        w5[k1*n + i0] = 1.0
        terms.append((u5, v5, w5))
        
        # M6
        u6 = np.zeros(n*m)
        v6 = np.zeros(m*p)
        w6 = np.zeros(p*n)
        u6[i1*m + j0] = 1.0
        u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0
        v6[j0*p + k1] = 1.0
        w6[k1*n + i1] = 1.0
        terms.append((u6, v6, w6))
        
        # M7
        u7 = np.zeros(n*m)
        v7 = np.zeros(m*p)
        w7 = np.zeros(p*n)
        u7[i0*m + j1] = 1.0
        u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0
        v7[j1*p + k1] = 1.0
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U.append(u)
            V.append(v)
            W.append(w)

    # Add Strassen blocks
    for I, J, K in blocks:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U.append(u)
                    V.append(v)
                    W.append(w)
                    
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    # Try to remove redundant terms
    def try_remove_term(U, V, W, idx):
        rank = U.shape[1]
        if rank <= 1:
            return False
        U_new = np.delete(U, idx, axis=1)
        V_new = np.delete(V, idx, axis=1)
        W_new = np.delete(W, idx, axis=1)
        return verify(U_new, V_new, W_new)
    
    # Attempt to remove terms one by one
    removed = True
    while removed and time.time() < deadline - 10:
        removed = False
        for idx in range(U_mat.shape[1]):
            if time.time() >= deadline - 10:
                break
            if try_remove_term(U_mat, V_mat, W_mat, idx):
                U_mat = np.delete(U_mat, idx, axis=1)
                V_mat = np.delete(V_mat, idx, axis=1)
                W_mat = np.delete(W_mat, idx, axis=1)
                removed = True
                break
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
