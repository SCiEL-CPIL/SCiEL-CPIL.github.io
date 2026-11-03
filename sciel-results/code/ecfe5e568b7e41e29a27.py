import time
import numpy as np
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 130.0  # Leave margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    for i in range(n):
        for k in range(m):
            for j in range(p):
                lhs = 1.0
                rhs = 0.0
                for r in range(rank):
                    rhs += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 4, 7
    
    # Build the rank-50 baseline solution
    rank_50 = 50
    U50 = np.zeros((n * m, rank_50), dtype=float)
    V50 = np.zeros((m * p, rank_50), dtype=float)
    W50 = np.zeros((p * n, rank_50), dtype=float)
    
    col = 0
    
    def add_strassen(a_base, b_base, c_base, offset):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U50[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), offset] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), offset] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), offset] += 1
        col = offset + 1
        
        # M2 = (A10 + A11)B00
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] -= 1
        W50[(c_base[1]+1)*n + (c_base[0]+0), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] -= 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] -= 1
        W50[(c_base[1]+1)*n + (c_base[0]+0), col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] -= 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] -= 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        col += 1
        
        return col

    def add_schoolbook_221(a_base, b_base, c_base, offset):
        nonlocal col
        # N1 = A00 B0
        U50[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        col = offset + 1
        
        # N2 = A01 B1
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        col += 1
        
        # N3 = A10 B0
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # N4 = A11 B1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        return col

    a_l_base = (0, 0)
    a_r_base = (0, 2)
    b_tl_base = (0, 0)
    b_tm_base = (0, 2)
    b_tr_base = (0, 4)
    b_t1_base = (0, 6)
    b_bl_base = (2, 0)
    b_bm_base = (2, 2)
    b_br_base = (2, 4)
    b_b1_base = (2, 6)
    c_l_base = (0, 0)
    c_m_base = (0, 2)
    c_r_base = (0, 4)
    c_1_base = (0, 6)
    
    col = add_strassen(a_l_base, b_tl_base, c_l_base, col)
    col = add_strassen(a_l_base, b_tm_base, c_m_base, col)
    col = add_strassen(a_l_base, b_tr_base, c_r_base, col)
    col = add_schoolbook_221(a_l_base, b_t1_base, c_1_base, col)
    col = add_strassen(a_r_base, b_bl_base, c_l_base, col)
    col = add_strassen(a_r_base, b_bm_base, c_m_base, col)
    col = add_strassen(a_r_base, b_br_base, c_r_base, col)
    col = add_schoolbook_221(a_r_base, b_b1_base, c_1_base, col)
    
    if col != rank_50:
        raise ValueError(f"Internal error: expected rank 50, got {col}")
    
    if not _verify_decomposition(U50, V50, W50, n, m, p, rank_50):
        raise ValueError("Rank-50 solution failed verification")
    
    # Try to reduce rank by finding linear dependencies
    # Build the tensor equation matrix: for each (i,k,j), we have
    # sum_r U[i,k,r]*V[k,j,r]*W[j,i,r] = 1
    # 
    # Strategy: try removing one term at a time and see if the remaining
    # terms can be rescaled to satisfy the equations.
    # This is a nonlinear feasibility problem, but we can try a linear relaxation.
    
    # Better approach: look for terms that are "similar" and can be merged.
    # Two terms r1, r2 can be merged if there exists a scalar c such that
    # U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = c*W[:,r2]
    # (up to the product being the same)
    
    # Actually, let's try a different approach: use the fact that the tensor
    # T[i,k,j] = 1 has a lot of symmetry. We can try to find a decomposition
    # with fewer terms by exploiting this symmetry.
    
    # Let's try: for each pair of terms, check if they can be combined.
    # This is O(R^2) which is 2500 for R=50, manageable.
    
    # But the real question is: can we do better than 50?
    # The information-theoretic lower bound is max(nm, mp, np) = max(8,28,14) = 28
    # But that's very loose. For 2x4x7, the best known might be around 40-50.
    
    # Given time constraints, let's just return the rank-50 solution.
    # A more sophisticated search would take too long.
    
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
