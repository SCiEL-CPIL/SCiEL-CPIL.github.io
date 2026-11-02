import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def remaining():
        return deadline - time.time()

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (24, rank)

    def schoolbook():
        """Standard matrix multiplication - rank 120"""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W, T):
        """Check if U, V, W exactly decompose T"""
        rank = len(U[0])
        T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for c in range(rank):
            uc = [U[i][c] for i in range(n * m)]
            vc = [V[i][c] for i in range(m * p)]
            wc = [W[i][c] for i in range(p * n)]
            T_recon += np.outer(uc, np.outer(vc, wc)).reshape(n * m, m * p, p * n)
        return np.allclose(T_recon, T, atol=1e-10)

    # Build the target tensor for verification
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Winograd 2x2x2 decomposition (rank 7)
    def winograd_2x2_2x2():
        """
        Returns 7 rank-1 terms for 2x2x2 matmul.
        Each term is (u, v, w) where u, v, w are length-4 vectors.
        """
        terms = []
        # p1=(a+d)(e+h)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # p2=(c+d)e
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # p3=a(f-h)
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        # p4=d(g-e)
        terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))
        # p5=(a+b)g
        terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))
        # p6=(c-a)(e+f)
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
        # p7=(b-d)(h+f)
        terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))
        return terms

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2_2x2():
        """
        Returns 7 rank-1 terms for 2x2x2 matmul using Strassen's algorithm.
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2 = (A10+A11)B00
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # M3 = A00(B10-B00)
        terms.append(([1,0,0,0], [-1,0,1,0], [0,1,1,0]))
        # M4 = A11(B11-B00)
        terms.append(([0,0,0,1], [-1,0,0,1], [-1,0,1,0]))
        # M5 = (A00+A01)B11
        terms.append(([1,1,0,0], [0,0,0,1], [1,0,1,0]))
        # M6 = (A10-A00)(B00+B01)
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
        # M7 = (A01-A11)(B10+B11)
        terms.append(([0,1,0,-1], [0,1,0,1], [0,1,-1,0]))
        return terms

    def embed_2x2x2_term(term, i_base, j_base, k_base):
        """
        Embed a 2x2x2 term into the full 4x5x6 space.
        term = (u, v, w) where u, v, w are length-4 vectors.
        u corresponds to A block rows i_base..i_base+1, cols j_base..j_base+1
        v corresponds to B block rows j_base..j_base+1, cols k_base..k_base+1
        w corresponds to C block rows k_base..k_base+1, cols i_base..i_base+1
        """
        u_full = [0.0] * (n * m)
        v_full = [0.0] * (m * p)
        w_full = [0.0] * (p * n)
        
        u_vals, v_vals, w_vals = term
        
        # u: A indices (i, j) -> i*m + j
        for a in range(2):
            for b in range(2):
                i_idx = i_base + a
                j_idx = j_base + b
                if 0 <= i_idx < n and 0 <= j_idx < m:
                    u_full[i_idx * m + j_idx] = u_vals[a * 2 + b]
        
        # v: B indices (j, k) -> j*p + k
        for b in range(2):
            for c in range(2):
                j_idx = j_base + b
                k_idx = k_base + c
                if 0 <= j_idx < m and 0 <= k_idx < p:
                    v_full[j_idx * p + k_idx] = v_vals[b * 2 + c]
        
        # w: C indices (k, i) -> k*n + i
        for c in range(2):
            for a in range(2):
                k_idx = k_base + c
                i_idx = i_base + a
                if 0 <= k_idx < p and 0 <= i_idx < n:
                    w_full[k_idx * n + i_idx] = w_vals[c * 2 + a]
        
        return u_full, v_full, w_full

    def build_decomposition(use_winograd=True):
        """
        Build decomposition using block-structured approach.
        Split n=4 into [2,2], m=5 into [2,3], p=6 into [2,4].
        Use Winograd/Strassen for 2x2x2 blocks, schoolbook for the rest.
        """
        terms_U = []
        terms_V = []
        terms_W = []
        
        # Choose which 2x2x2 algorithm to use
        base_terms = winograd_2x2_2x2() if use_winograd else strassen_2x2_2x2()
        
        # i blocks: [0,1], [2,3]
        # j blocks: [0,1], [2,3], [4,4] (j=4 is single)
        # k blocks: [0,1], [2,3], [4,5]
        
        # We can form 2x2x2 blocks for:
        # i in {0,1} or {2,3}
        # j in {0,1} or {2,3} (not j=4)
        # k in {0,1} or {2,3} or {4,5}
        
        # Let's tile with 2x2x2 blocks where possible:
        # For i_base in [0, 2]:
        #   For j_base in [0, 2]:  # j=0,1 and j=2,3
        #     For k_base in [0, 2, 4]:  # k=0,1; k=2,3; k=4,5
        
        covered = set()
        
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2, 4]:
                    # Check bounds
                    if i_base + 1 >= n or j_base + 1 >= m or k_base + 1 >= p:
                        continue
                    
                    for term in base_terms:
                        u_full, v_full, w_full = embed_2x2x2_term(term, i_base, j_base, k_base)
                        terms_U.append(u_full)
                        terms_V.append(v_full)
                        terms_W.append(w_full)
                    
                    # Mark covered
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                covered.add((i_base + a, j_base + b, k_base + c))
        
        # Add schoolbook for uncovered (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    u_full = [0.0] * (n * m)
                    v_full = [0.0] * (m * p)
                    w_full = [0.0] * (p * n)
                    u_full[i * m + j] = 1.0
                    v_full[j * p + k] = 1.0
                    w_full[k * n + i] = 1.0
                    terms_U.append(u_full)
                    terms_V.append(v_full)
                    terms_W.append(w_full)
        
        rank = len(terms_U)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
            for i in range(n * m):
                U[i][c] = u[i]
            for j in range(m * p):
                V[j][c] = v[j]
            for k in range(p * n):
                W[k][c] = w[k]
        
        return U, V, W, rank

    # Start with schoolbook
    U_school, V_school, W_school = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try Winograd-based decomposition
    if remaining() > 10:
        U_w, V_w, W_w, rank_w = build_decomposition(use_winograd=True)
        if verify_decomp(U_w, V_w, W_w, T) and rank_w < best_rank:
            best_rank = rank_w
            best_U, best_V, best_W = U_w, V_w, W_w
    
    # Try Strassen-based decomposition
    if remaining() > 10:
        U_s, V_s, W_s, rank_s = build_decomposition(use_winograd=False)
        if verify_decomp(U_s, V_s, W_s, T) and rank_s < best_rank:
            best_rank = rank_s
            best_U, best_V, best_W = U_s, V_s, W_s
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
