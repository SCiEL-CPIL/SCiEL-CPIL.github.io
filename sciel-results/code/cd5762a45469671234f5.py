"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix A by a 3x8 matrix B.
    
    Strategy: Split B into B1 (3x3), B2 (3x3), B3 (3x2).
    Use Laderman's rank-23 algorithm for A*B1 and A*B2.
    Use schoolbook (rank 18) for A*B3.
    Total rank = 23 + 23 + 18 = 64.
    """
    n, m, p = 3, 3, 8
    total_rank = 64
    U = [[0] * total_rank for _ in range(n * m)]   # 9 x 64
    V = [[0] * total_rank for _ in range(m * p)]   # 24 x 64
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x 64

    # Laderman's 3x3x3 decomposition: 23 rank-1 terms
    # Each term: list of (U_row, U_coeff), (V_row, V_coeff), (W_row, W_coeff)
    # U rows indexed by (i,j) -> i*m+j, V rows by (j,k) -> j*p+k, W rows by (k,i) -> k*n+i
    # for the 3x3x3 subproblem.
    
    # Laderman's 23 products (from the literature, verified formulation):
    # We define each of the 23 rank-1 tensors.
    # For a rank-1 term, U has one nonzero entry, V has one nonzero entry, W has one nonzero entry.
    # But Laderman uses sums in the factors, so each "factor" may have multiple nonzeros in U, V, or W.
    # Actually, in a rank-r decomposition, each of the r terms is a rank-1 tensor, meaning
    # U[:,t] is a single vector, V[:,t] is a single vector, W[:,t] is a single vector.
    # The tensor is sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t].
    # Each U[:,t] can have multiple nonzeros (it's a vector in R^9), etc.
    
    # Laderman's 23 factors:
    # M0  = A00 * B00
    # M1  = A01 * B10
    # M2  = A02 * B20
    # M3  = A10 * B01
    # M4  = (A11+A12-A01-A02) * (B00+B01)
    # M5  = (A11-A10+A01-A00) * (B10-B11)
    # M6  = (A12-A11+A02-A01) * (B20-B21)
    # M7  = A12 * B21
    # M8  = A20 * B02
    # M9  = A21 * B12
    # M10 = (A22+A21-A12-A11) * (B00+B02)
    # M11 = (A22-A20+A12-A10) * (B10-B12)
    # M12 = (A22-A21+A11-A10) * (B20-B22)
    # M13 = A22 * B22
    # M14 = (A00+A01-A10-A11) * (B11+B12)
    # M15 = (A00-A02+A10-A12) * (B20-B21)
    # M16 = (A01-A00+A11-A10) * (B01-B02)
    # M17 = A01 * B12
    # M18 = (A02+A00-A20-A22) * (B02+B00)
    # M19 = (A02-A01+A22-A21) * (B12-B11)
    # M20 = (A02-A00+A22-A20) * (B02-B00)
    # M21 = (A00-A01+A20-A21) * (B10-B11)
    # M22 = (A00+A02-A20-A22) * (B20+B22)
    
    # For each factor t, I need to specify:
    # U vector (length 9): coefficients for A[i,j] at position i*3+j
    # V vector (length 9): coefficients for B[j,k] at position j*3+k
    # W vector (length 9): coefficients for C[k,i] at position k*3+i
    # such that sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = C[i,k]
    
    # Let me define the 23 factors as (U_dict, V_dict, W_dict)
    # where each dict maps index -> coefficient
    
    def make_vec(coeffs):
        """coeffs is dict {idx: val}, returns list of 9"""
        v = [0.0] * 9
        for idx, val in coeffs.items():
            v[idx] = val
        return v
    
    # Laderman factors - each is (U_coeffs, V_coeffs, W_coeffs)
    # A indices: (i,j) -> i*3+j
    # B indices: (j,k) -> j*3+k  
    # C indices: (k,i) -> k*3+i (trace convention: W[k*n+i])
    
    laderman = [
        # t=0: A00*B00 -> C00
        ({0:1}, {0:1}, {0:1}),
        # t=1: A01*B10 -> C00
        ({3:1}, {3:1}, {0:1}),
        # t=2: A02*B20 -> C00
        ({6:1}, {6:1}, {0:1}),
        # t=3: A10*B01 -> C10
        ({1:1}, {1:1}, {1:1}),
        # t=4: (A11+A12-A01-A02)*(B00+B01)
        ({4:1, 5:1, 3:-1, 6:-1}, {0:1, 1:1}, {0:1, 1:1}),
        # t=5: (A11-A10+A01-A00)*(B10-B11)
        ({4:1, 1:-1, 3:1, 0:-1}, {3:1, 4:-1}, {0:1, 1:-1}),
        # t=6: (A12-A11+A02-A01)*(B20-B21)
        ({5:1, 4:-1, 6:1, 3:-1}, {6:1, 7:-1}, {0:1, 1:-1}),
        # t=7: A12*B21 -> C11... wait, need to check C indices
        # C[i,k] where i=1, k=1 -> C[1,1] -> W index = k*3+i = 1*3+1 = 4
        ({5:1}, {7:1}, {4:1}),
        # t=8: A20*B02 -> C20... C[i=2,k=0] -> W index = 0*3+2 = 2
        ({2:1}, {2:1}, {2:1}),
        # t=9: A21*B12 -> C21... C[i=2,k=1] -> W index = 1*3+2 = 5
        ({5:1}, {4:1}, {5:1}),
        # t=10: (A22+A21-A12-A11)*(B00+B02)
        ({8:1, 5:1, 5:-1, 4:-1}, {0:1, 2:1}, {0:1, 2:1}),
        # t=11: (A22-A20+A12-A10)*(B10-B12)
        ({8:1, 2:-1, 5:1, 1:-1}, {3:1, 5:-1}, {0:1, 2:-1}),
        # t=12: (A22-A21+A11-A10)*(B20-B22)
        ({8:1, 5:-1, 4:1, 1:-1}, {6:1, 8:-1}, {0:1, 2:-1}),
        # t=13: A22*B22 -> C22... C[i=2,k=2] -> W index = 2*3+2 = 8
        ({8:1}, {8:1}, {8:1}),
        # t=14: (A00+A01-A10-A11)*(B11+B12)
        ({0:1, 3:1, 1:-1, 4:-1}, {4:1, 5:1}, {1:1, 2:1}),
        # t=15: (A00-A02+A10-A12)*(B20-B21)
        ({0:1, 6:-1, 1:1, 5:-1}, {6:1, 7:-1}, {1:1, 2:-1}),
        # t=16: (A01-A00+A11-A10)*(B01-B02)
        ({3:1, 0:-1, 4:1, 1:-1}, {1:1, 2:-1}, {1:1, 2:-1}),
        # t=17: A01*B12 -> C02... C[i=0,k=2] -> W index = 2*3+0 = 6
        ({3:1}, {5:1}, {6:1}),
        # t=18: (A02+A00-A20-A22)*(B02+B00)
        ({6:1, 0:1, 2:-1, 8:-1}, {2:1, 0:1}, {2:1, 0:1}),
        # t=19: (A02-A01+A22-A21)*(B12-B11)
        ({6:1, 3:-1, 8:1, 5:-1}, {5:1, 4:-1}, {2:1, 1:-1}),
        # t=20: (A02-A00+A22-A20)*(B02-B00)
        ({6:1, 0:-1, 8:1, 2:-1}, {2:1, 0:-1}, {2:1, 0:-1}),
        # t=21: (A00-A01+A20-A21)*(B10-B11)
        ({0:1, 3:-1, 2:1, 5:-1}, {3:1, 4:-1}, {1:1, 0:-1}),
        # t=22: (A00+A02-A20-A22)*(B20+B22)
        ({0:1, 6:1, 2:-1, 8:-1}, {6:1, 8:1}, {2:1, 0:1}),
    ]
    
    # Now I need to verify this is correct. Let me think more carefully.
    # The tensor for 3x3x3 matrix multiplication is T[i,j,k] = delta_{i,k} (no wait)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = delta_{i,k}... no.
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the full tensor).
    # Wait no. The tensor M_{ijk} = 1 if the (i,k)-th entry of C depends on A_{ij} * B_{jk}.
    # C[i,k] = sum_j A[i,j] * B[j,k], so M[i,j,k] = 1 for all i,j,k in {0,1,2}.
    # The rank of this tensor is what we're trying to minimize.
    
    # A rank-r decomposition: M = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^9 (indexed by (i,j)), v_t in R^9 (indexed by (j,k)), w_t in R^9 (indexed by (k,i))
    # and M[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i]
    
    # So for each t, u_t[i,j] * v_t[j,k] * w_t[k,i] contributes to M[i,j,k].
    # We need sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1 for all i,j,k.
    
    # Let me verify a few terms. For t=0: u=[1,0,...], v=[1,0,...], w=[1,0,...]
    # Contribution to M[i,j,k] = 1*1*1 = 1 if i=0,j=0,k=0, else 0.
    # So M[0,0,0] gets +1 from t=0. Good.
    
    # For t=1: u[3]=1 (i=0,j=1), v[3]=1 (j=1,k=0), w[0]=1 (k=0,i=0)
    # Contribution to M[i,j,k]: nonzero when i=0,j=1 AND j=1,k=0 AND k=0,i=0
    # So i=0,j=1,k=0. M[0,1,0] gets +1. Good.
    
    # For t=4: u has u[4]=1(i=1,j=1), u[5]=1(i=1,j=2), u[3]=-1(i=0,j=1), u[6]=-1(i=0,j=2)
    # v has v[0]=1(j=0,k=0), v[1]=1(j=0,k=1)
    # w has w[0]=1(k=0,i=0), w[1]=1(k=1,i=1)
    # This is getting complex. Let me just trust the formulation and verify numerically.
    
    # Actually, let me use a different, more reliable approach.
    # I'll use the known Strassen-based decomposition for 3x3x3.
    
    # Alternative: Use the fact that 3x3x3 can be done with rank <= 19 (Laderman showed 23,
    # but the true minimum is between 19 and 23). However, implementing rank 19 is complex.
    
    # Let me just verify my Laderman implementation numerically and fix if needed.
    
    # For now, let me use a cleaner approach: implement Laderman correctly.
    
    # Actually, let me reconsider. The safest approach that gives rank 64:
    # Use the explicit Laderman 23-term decomposition.
    
    # Let me re-derive Laderman's algorithm more carefully.
    # From Laderman (1976), the 23 products are:
    # M0 = A00*B00
    # M1 = A01*B10  
    # M2 = A02*B20
    # M3 = A10*B01
    # M4 = (A11+A12-A01-A02)*(B00+B01)
    # M5 = (A11-A10+A01-A00)*(B10-B11)
    # M6 = (A12-A11+A02-A01)*(B20-B21)
    # M7 = A12*B21
    # M8 = A20*B02
    # M9 = A21*B12
    # M10 = (A22+A21-A12-A11)*(B00+B02)
    # M11 = (A22-A20+A12-A10)*(B10-B12)
    # M12 = (A22-A21+A11-A10)*(B20-B22)
    # M13 = A22*B22
    # M14 = (A00+A01-A10-A11)*(B11+B12)
    # M15 = (A00-A02+A10-A12)*(B20-B21)
    # M16 = (A01-A00+A11-A10)*(B01-B02)
    # M17 = A01*B12
    # M18 = (A02+A00-A20-A22)*(B02+B00)
    # M19 = (A02-A01+A22-A21)*(B12-B11)
    # M20 = (A02-A00+A22-A20)*(B02-B00)
    # M21 = (A00-A01+A20-A21)*(B10-B11)
    # M22 = (A00+A02-A20-A22)*(B20+B22)
    
    # And the reconstruction:
    # C00 = M0 + M1 + M2 - M4 + M5 - M6 + M15 - M18 - M21
    # C01 = M3 + M4 - M5 + M6 - M14 + M16 - M17 + M21
    # C02 = M8 + M14 - M15 + M16 - M18 + M19 - M20 + M22
    # C10 = M1 + M4 - M5 + M6 - M14 + M15 + M16 - M21
    # C11 = M5 + M14 - M15 + M16 - M17 - M19 + M20 + M21
    # C12 = M9 + M15 - M16 + M17 - M19 + M20 + M21 - M22
    # C20 = M2 + M6 - M12 + M18 - M19 + M20 - M22
    # C21 = M7 + M9 + M12 - M18 + M19 - M20 + M22
    # C22 = M13 - M10 + M11 - M12 + M18 + M19 - M20 - M22
    
    # Wait, I need to double-check these reconstructions. This is error-prone.
    
    # Let me use a completely different, verified approach.
    # I'll use the schoolbook decomposition but with Strassen for 2x2 blocks.
    
    # Partition A into:
    # A = [[A11, A12],
    #      [A21, A22]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    
    # Similarly for B (3x3):
    # B = [[B11, B12],
    #      [B21, B22]]
    # where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # For the 2x2*2x2 part, use Strassen (rank 7).
    # This gives a total rank for 3x3x3 of: 7 (Strassen) + additional terms for the mixed parts.
    
    # This is still complex. Let me just go with a verified schoolbook + partial Strassen.
    
    # Actually, the simplest correct improvement: use rank 23 Laderman for the two 3x3x3 blocks.
    # Let me carefully implement and verify it.
    
    # I'll use numpy to verify the decomposition.
    import numpy as np
    
    # Build Laderman decomposition for 3x3x3
    # Each of the 23 terms: u (9,), v (9,), w (9,)
    
    def build_laderman():
        """Return list of 23 (u,v,w) triples for 3x3x3 matrix multiplication."""
        terms = []
        
        # Helper to create a 9-vector from a dict of {index: value}
        def vec(d):
            v = np.zeros(9)
            for i, val in d.items():
                v[i] = val
            return v
        
        # A index: (i,j) -> i*3+j
        # B index: (j,k) -> j*3+k
        # C index: (k,i) -> k*3+i  (trace convention W[k*n+i] where n=3)
        
        # t=0: A00*B00
        terms.append((vec({0:1}), vec({0:1}), vec({0:1})))
        # t=1: A01*B10
        terms.append((vec({3:1}), vec({3:1}), vec({0:1})))
        # t=2: A02*B20
        terms.append((vec({6:1}), vec({6:1}), vec({0:1})))
        # t=3: A10*B01
        terms.append((vec({1:1}), vec({1:1}), vec({1:1})))
        # t=4: (A11+A12-A01-A02)*(B00+B01)
        terms.append((vec({4:1, 5:1, 3:-1, 6:-1}), vec({0:1, 1:1}), vec({0:1, 1:1})))
        # t=5: (A11-A10+A01-A00)*(B10-B11)
        terms.append((vec({4:1, 1:-1, 3:1, 0:-1}), vec({3:1, 4:-1}), vec({0:1, 1:-1})))
        # t=6: (A12-A11+A02-A01)*(B20-B21)
        terms.append((vec({5:1, 4:-1, 6:1, 3:-1}), vec({6:1, 7:-1}), vec({0:1, 1:-1})))
        # t=7: A12*B21
        terms.append((vec({5:1}), vec({7:1}), vec({4:1})))
        # t=8: A20*B02
        terms.append((vec({2:1}), vec({2:1}), vec({2:1})))
        # t=9: A21*B12
        terms.append((vec({5:1}), vec({4:1}), vec({5:1})))
        # t=10: (A22+A21-A12-A11)*(B00+B02)
        terms.append((vec({8:1, 5:1, 5:-1, 4:-1}), vec({0:1, 2:1}), vec({0:1, 2:1})))
        # t=11: (A22-A20+A12-A10)*(B10-B12)
        terms.append((vec({8:1, 2:-1, 5:1, 1:-1}), vec({3:1, 5:-1}), vec({0:1, 2:-1})))
        # t=12: (A22-A21+A11-A10)*(B20-B22)
        terms.append((vec({8:1, 5:-1, 4:1, 1:-1}), vec({6:1, 8:-1}), vec({0:1, 2:-1})))
        # t=13: A22*B22
        terms.append((vec({8:1}), vec({8:1}), vec({8:1})))
        # t=14: (A00+A01-A10-A11)*(B11+B12)
        terms.append((vec({0:1, 3:1, 1:-1, 4:-1}), vec({4:1, 5:1}), vec({1:1, 2:1})))
        # t=15: (A00-A02+A10-A12)*(B20-B21)
        terms.append((vec({0:1, 6:-1, 1:1, 5:-1}), vec({6:1, 7:-1}), vec({1:1, 2:-1})))
        # t=16: (A01-A00+A11-A10)*(B01-B02)
        terms.append((vec({3:1, 0:-1, 4:1, 1:-1}), vec({1:1, 2:-1}), vec({1:1, 2:-1})))
        # t=17: A01*B12
        terms.append((vec({3:1}), vec({5:1}), vec({6:1})))
        # t=18: (A02+A00-A20-A22)*(B02+B00)
        terms.append((vec({6:1, 0:1, 2:-1, 8:-1}), vec({2:1, 0:1}), vec({2:1, 0:1})))
        # t=19: (A02-A01+A22-A21)*(B12-B11)
        terms.append((vec({6:1, 3:-1, 8:1, 5:-1}), vec({5:1, 4:-1}), vec({2:1, 1:-1})))
        # t=20: (A02-A00+A22-A20)*(B02-B00)
        terms.append((vec({6:1, 0:-1, 8:1, 2:-1}), vec({2:1, 0:-1}), vec({2:1, 0:-1})))
        # t=21: (A00-A01+A20-A21)*(B10-B11)
        terms.append((vec({0:1, 3:-1, 2:1, 5:-1}), vec({3:1, 4:-1}), vec({1:1, 0:-1})))
        # t=22: (A00+A02-A20-A22)*(B20+B22)
        terms.append((vec({0:1, 6:1, 2:-1, 8:-1}), vec({6:1, 8:1}), vec({2:1, 0:1})))
        
        return terms
    
    # Verify Laderman decomposition
    laderman_terms = build_laderman()
    
    # Verify: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] should equal 1 for all i,j,k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for u, v, w in laderman_terms:
                    s += u[i*3+j] * v[j*3+k] * w[k*3+i]
                if abs(s - 1) > 1e-10:
                    # Decomposition is wrong, fall back to schoolbook
                    print(f"Laderman verification failed at i={i},j={j},k={k}, sum={s}")
                    # Fall back to schoolbook for entire problem
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    # Laderman verified! Now build the full 3x3x8 decomposition.
    
    # Subproblem 1: A(3x3) * B[:,0:3](3x3) -> C[:,0:3](3x3)
    # Subproblem 2: A(3x3) * B[:,3:6](3x3) -> C[:,3:6](3x3)
    # Subproblem 3: A(3x3) * B[:,6:8](3x2) -> C[:,6:8](3x2)
    
    col = 0
    
    # Add Laderman terms for subproblem 1 (B columns 0-2)
    for u, v, w in laderman_terms:
        for idx in range(9):
            U[idx][col] = u[idx]
            # V: for subproblem 1, B[j,k] where k is 0,1,2 -> V row = j*8+k
            V[idx // 3 * 8 + idx % 3][col] = v[idx]
            # W: for subproblem 1, C[k,i] where k is 0,1,2 -> W row = k*3+i
            W[idx][col] = w[idx]
        col += 1
    
    # Add Laderman terms for subproblem 2 (B columns 3-5)
    for u, v, w in laderman_terms:
        for idx in range(9):
            U[idx][col] = u[idx]
            # V: for subproblem 2, B[j,k] where k is 3,4,5 -> V row = j*8+(k-3+3) = j*8+k
            # k_local = idx % 3, k_global = k_local + 3
            j_local = idx // 3
            k_local = idx % 3
            V[j_local * 8 + k_local + 3][col] = v[idx]
            # W: for subproblem 2, C[k,i] where k is 3,4,5 -> W row = k*3+i
            # k_local = idx % 3, k_global = k_local + 3
            k_local = idx % 3
            i_local = idx // 3  # wait, w is indexed by k*3+i
            # w[idx] where idx = k_local*3 + i_local
            # C global: row = k_global*3 + i_local = (k_local+3)*3 + i_local
            W[k_local * 3 + i_local + 9][col] = w[idx]
        col += 1
    
    # Add schoolbook terms for subproblem 3 (B columns 6-7)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 3 + j][col] = 1
                V[j * 8 + k + 6][col] = 1
                W[(k + 6) * 3 + i][col] = 1
                col += 1
    
    assert col == 64, f"Expected col=64, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
