"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 145
    
    def check_decomposition(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0
                    actual = sum(U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r] 
                                  for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def build_rank17():
        """
        Rank-17 decomposition: 8 terms for cols 0-2 (generalized Strassen k=3)
        + 7 terms for cols 3-4 (standard Strassen k=2)
        """
        rank = 17
        U = [[0] * rank for _ in range(n * m)]   # (4, 17)
        V = [[0] * rank for _ in range(m * p)]   # (10, 17)
        W = [[0] * rank for _ in range(p * n)]   # (10, 17)
        
        # Generalized Strassen for 2x2 * 2x3 (columns c, c+1, c+2)
        def add_strassen_k3(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+2)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+2][r] = 1
            W[c*n+0][r] = 1; W[(c+2)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+2)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            # M5: f=A00+A01, g=B1(c+2)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+2)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
            
            # M8: f=A01, g=B1(c+2)
            r = base + 7
            U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[(c+2)*n+0][r] = 1
        
        # Standard Strassen for 2x2 * 2x2 (columns c, c+1)
        def add_strassen_k2(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+1)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[c*n+1][r] = 1
            
            # M5: f=A00+A01, g=B1(c+1)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
        
        # Columns 0,1,2: generalized Strassen (8 terms)
        add_strassen_k3(0, 0)
        
        # Columns 3,4: standard Strassen (7 terms)
        add_strassen_k2(3, 8)
        
        return U, V, W
    
    def build_rank18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W
    
    def try_rank17_als(U_init, V_init, W_init, rank, max_iters=50, tol=1e-10):
        """
        ALS (Alternating Least Squares) to refine a rank-17 decomposition.
        Treats U, V, W as variables and minimizes the tensor approximation error.
        """
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (a,b), solve for U[a*m+b, :]
            for a in range(n):
                for b in range(m):
                    # Residual for entry (a,b,c) across all c
                    # We want: sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] = 1 for all c
                    # This is linear in U[a*m+b, :]
                    # Let's build the system: for each c, sum_r U[a*m+b, r] * (V[b*p+c, r] * W[c*n+a, r]) = 1
                    # This gives p equations in rank unknowns
                    if rank > p:
                        # Underdetermined, use least squares
                        A_mat = np.zeros((p, rank))
                        for c in range(p):
                            for r in range(rank):
                                A_mat[c, r] = V[b*p+c, r] * W[c*n+a, r]
                        b_vec = np.ones(p)
                        # Solve min ||A_mat @ x - b_vec||^2
                        try:
                            sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[a*m+b, :] = sol
                        except:
                            pass
                    else:
                        A_mat = np.zeros((p, rank))
                        for c in range(p):
                            for r in range(rank):
                                A_mat[c, r] = V[b*p+c, r] * W[c*n+a, r]
                        b_vec = np.ones(p)
                        try:
                            U[a*m+b, :] = np.linalg.solve(A_mat, b_vec)
                        except:
                            try:
                                U[a*m+b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            except:
                                pass
            
            # Update V: for each (b,c), solve for V[b*p+c, :]
            for b in range(m):
                for c in range(p):
                    A_mat = np.zeros((n, rank))
                    for a in range(n):
                        for r in range(rank):
                            A_mat[a, r] = U[a*m+b, r] * W[c*n+a, r]
                    b_vec = np.ones(n)
                    try:
                        V[b*p+c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (c,a), solve for W[c*n+a, :]
            for c in range(p):
                for a in range(n):
                    A_mat = np.zeros((m, rank))
                    for b in range(m):
                        for r in range(rank):
                            A_mat[b, r] = U[a*m+b, r] * V[b*p+c, r]
                    b_vec = np.ones(m)
                    try:
                        W[c*n+a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = 0.0
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        actual = sum(U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] for r in range(rank))
                        err += (actual - 1.0)**2
            if err < tol:
                break
        
        # Convert to list of lists and check
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if check_decomposition(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list
        return None
    
    # Try rank 17 first
    U17, V17, W17 = build_rank17()
    if check_decomposition(U17, V17, W17, 17):
        return {"U": U17, "V": V17, "W": W17}
    
    # Try ALS refinement on rank 17
    if time.time() < deadline - 10:
        result = try_rank17_als(U17, V17, W17, 17)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to rank 18
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
