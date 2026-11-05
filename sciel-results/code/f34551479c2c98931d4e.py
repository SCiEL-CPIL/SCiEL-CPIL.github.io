"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(T_reconstructed, 1.0, atol=1e-9):
            return False
        return True

    # Schoolbook decomposition: rank = n*m*p = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strassen-based decomposition: rank = 36
    rank_strassen = 36
    U = [[0] * rank_strassen for _ in range(n * m)]
    V = [[0] * rank_strassen for _ in range(m * p)]
    W = [[0] * rank_strassen for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen 2x2x2 coefficients
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Standard 2x2x1 coefficients
    standard_terms_2x2x1 = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]

    # 2x2x3 decomposition with rank 9
    # B indices for 2x2x3: 0:(j=0,k=0), 1:(j=0,k=1), 2:(j=0,k=2), 3:(j=1,k=0), 4:(j=1,k=1), 5:(j=1,k=2)
    # C indices for 2x2x3: 0:(i=0,k=0), 1:(i=0,k=1), 2:(i=0,k=2), 3:(i=1,k=0), 4:(i=1,k=1), 5:(i=1,k=2)
    # A indices: 0:(i=0,j=0), 1:(i=0,j=1), 2:(i=1,j=0), 3:(i=1,j=1)
    
    # Strassen terms for cols 0,1 (B indices 0,1,3,4; C indices 0,1,3,4)
    strassen_2x2x3 = [
        ({0:1, 3:1}, {0:1, 4:1}, {0:1, 4:1}),   # (A00+A11)(B00+B11) -> C00+1, C11+1
        ({2:1, 3:1}, {0:1}, {3:1, 4:-1}),        # (A10+A11)B00 -> C10+1, C11-1
        ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),        # A00(B01-B11) -> C01+1, C11+1
        ({3:1}, {3:1, 0:-1}, {0:1, 3:1}),        # A11(B10-B00) -> C00+1, C10+1
        ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),        # (A00+A01)B11 -> C00-1, C01+1
        ({2:1, 0:-1}, {0:1, 1:1}, {4:1}),        # (A10-A00)(B00+B01) -> C11+1
        ({1:1, 3:-1}, {3:1, 4:1}, {0:1})         # (A01-A11)(B10+B11) -> C00+1
    ]
    
    # Extra terms for col 2 (B indices 2,5; C indices 2,5)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    extra_2x2x3 = [
        ({0:1, 2:1}, {2:1}, {2:1, 5:1}),   # (A00+A10)*B02 -> C02+=A00*B02, C12+=A10*B02
        ({1:1, 3:1}, {5:1}, {2:1, 5:1})    # (A01+A11)*B12 -> C02+=A01*B12, C12+=A11*B12
    ]
    
    terms_2x2x3 = strassen_2x2x3 + extra_2x2x3
    
    # Verify 2x2x3 decomposition
    def verify_2x2x3(terms):
        rank = len(terms)
        U_l = np.zeros((2, 2, rank))
        V_l = np.zeros((2, 3, rank))
        W_l = np.zeros((3, 2, rank))
        for r, (u_t, v_t, w_t) in enumerate(terms):
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                U_l[li, lj, r] += val
            for idx, val in v_t.items():
                lj, lk = idx // 3, idx % 3
                V_l[lj, lk, r] += val
            for idx, val in w_t.items():
                lk, li = idx // 2, idx % 2
                W_l[lk, li, r] += val
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_l, V_l, W_l)
        T_target = np.ones((2, 2, 3))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    if not verify_2x2x3(terms_2x2x3):
        # Fallback: use 11 terms (7 Strassen + 4 standard for col 2)
        standard_col2 = [
            ({0:1}, {2:1}, {2:1}),   # A00*B02 -> C02
            ({1:1}, {5:1}, {2:1}),   # A01*B12 -> C02
            ({2:1}, {2:1}, {5:1}),   # A10*B02 -> C12
            ({3:1}, {5:1}, {5:1})    # A11*B12 -> C12
        ]
        terms_2x2x3 = strassen_2x2x3 + standard_col2

    # Block decomposition
    # m=4 split into two blocks of 2. q in {0, 1}
    # p=5 split into one block of 2 (s=0, k=0,1) and one block of 3 (s=1, k=2,3,4)
    
    total_rank = 2 * (7 + len(terms_2x2x3))
    if total_rank < rank_strassen:
        # Reallocate with smaller rank
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
    
    for q in range(2):
        for s in range(2):
            if s == 0:
                # 2x2x2 Strassen, k_global = 0,1
                for u_t, v_t, w_t in strassen_terms:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_global[li * m + (2*q + lj)] = val
                    for idx, val in v_t.items():
                        lj, lk = idx // 2, idx % 2
                        v_global[(2*q + lj) * p + lk] = val
                    for idx, val in w_t.items():
                        lk, li = idx // 2, idx % 2
                        w_global[lk * n + li] = val
                    add_term(u_global, v_global, w_global)
            else:
                # 2x2x3, k_global = 2,3,4
                for u_t, v_t, w_t in terms_2x2x3:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_global[li * m + (2*q + lj)] = val
                    for idx, val in v_t.items():
                        # B index in 2x2x3: 0:(j=0,k=0), 1:(j=0,k=1), 2:(j=0,k=2), 3:(j=1,k=0), 4:(j=1,k=1), 5:(j=1,k=2)
                        lj = idx // 3
                        lk = idx % 3
                        v_global[(2*q + lj) * p + (2 + lk)] = val
                    for idx, val in w_t.items():
                        # C index in 2x2x3: 0:(i=0,k=0), 1:(i=0,k=1), 2:(i=0,k=2), 3:(i=1,k=0), 4:(i=1,k=1), 5:(i=1,k=2)
                        lk = idx // 2
                        li = idx % 2
                        w_global[(2 + lk) * n + li] = val
                    add_term(u_global, v_global, w_global)

    # Verify the decomposition
    if col == total_rank and verify(U, V, W, total_rank):
        best = {"U": U, "V": V, "W": W}
    else:
        best = {"U": U_school, "V": V_school, "W": W_school}

    return best
# EVOLVE-BLOCK-END
