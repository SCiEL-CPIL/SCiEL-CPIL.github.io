import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 6
    
    # Rank-23 algorithm for 3x3 matrix multiplication
    # Terms are (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, coeff) for A
    # v_coeffs: list of (j, k, coeff) for B
    # w_coeffs: list of (k, i, coeff) for C
    
    def get_rank23_terms():
        terms = []
        # 1. a11 b11
        terms.append(([(0,0,1)], [(0,0,1)], [(0,0,1)]))
        # 2. a22 b22
        terms.append(([(1,1,1)], [(1,1,1)], [(1,1,1)]))
        # 3. a33 b33
        terms.append(([(2,2,1)], [(2,2,1)], [(2,2,1)]))
        # 4. (a11+a12)(b11+b12)
        terms.append(([(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)]))
        # 5. (a11+a13)(b11+b13)
        terms.append(([(0,0,1), (0,2,1)], [(0,0,1), (0,2,1)], [(0,0,1), (0,2,1)]))
        # 6. (a21+a22)(b21+b22)
        terms.append(([(1,0,1), (1,1,1)], [(1,0,1), (1,1,1)], [(1,0,1), (1,1,1)]))
        # 7. (a22+a23)(b22+b23)
        terms.append(([(1,1,1), (1,2,1)], [(1,1,1), (1,2,1)], [(1,1,1), (1,2,1)]))
        # 8. (a31+a32)(b31+b32)
        terms.append(([(2,0,1), (2,1,1)], [(2,0,1), (2,1,1)], [(2,0,1), (2,1,1)]))
        # 9. (a32+a33)(b32+b33)
        terms.append(([(2,1,1), (2,2,1)], [(2,1,1), (2,2,1)], [(2,1,1), (2,2,1)]))
        # 10. (a11+a21)(b11+b21)
        terms.append(([(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]))
        # 11. (a12+a22)(b12+b22)
        terms.append(([(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)]))
        # 12. (a13+a23)(b13+b23)
        terms.append(([(0,2,1), (1,2,1)], [(0,2,1), (1,2,1)], [(0,2,1), (1,2,1)]))
        # 13. (a11+a31)(b11+b31)
        terms.append(([(0,0,1), (2,0,1)], [(0,0,1), (2,0,1)], [(0,0,1), (2,0,1)]))
        # 14. (a22+a32)(b22+b32)
        terms.append(([(1,1,1), (2,1,1)], [(1,1,1), (2,1,1)], [(1,1,1), (2,1,1)]))
        # 15. (a23+a33)(b23+b33)
        terms.append(([(1,2,1), (2,2,1)], [(1,2,1), (2,2,1)], [(1,2,1), (2,2,1)]))
        # 16. (a11+a22+a33)(b11+b22+b33)
        terms.append(([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]))
        # 17. (a11+a22-a33)(b11+b22-b33)
        terms.append(([(0,0,1), (1,1,1), (2,2,-1)], [(0,0,1), (1,1,1), (2,2,-1)], [(0,0,1), (1,1,1), (2,2,-1)]))
        # 18. (a11-a22+a33)(b11-b22+b33)
        terms.append(([(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,-1), (2,2,1)]))
        # 19. (-a11+a22+a33)(-b11+b22+b33)
        terms.append(([(0,0,-1), (1,1,1), (2,2,1)], [(0,0,-1), (1,1,1), (2,2,1)], [(0,0,-1), (1,1,1), (2,2,1)]))
        # 20. (a12+a23)(b12+b23)
        terms.append(([(0,1,1), (1,2,1)], [(0,1,1), (1,2,1)], [(0,1,1), (1,2,1)]))
        # 21. (a13+a21)(b13+b21)
        terms.append(([(0,2,1), (1,0,1)], [(0,2,1), (1,0,1)], [(0,2,1), (1,0,1)]))
        # 22. (a21+a32)(b21+b32)
        terms.append(([(1,0,1), (2,1,1)], [(1,0,1), (2,1,1)], [(1,0,1), (2,1,1)]))
        # 23. (a31+a12)(b31+b12)
        terms.append(([(2,0,1), (0,1,1)], [(2,0,1), (0,1,1)], [(2,0,1), (0,1,1)]))
        
        return terms

    terms = get_rank23_terms()
    rank_half = len(terms)
    total_rank = 2 * rank_half
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill first half (B1, C1)
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for (i, j, coeff) in u_c:
            U[i*m+j, t] += coeff
        for (j, k, coeff) in v_c:
            V[j*p+k, t] += coeff
        for (k, i, coeff) in w_c:
            W[k*n+i, t] += coeff
            
    # Fill second half (B2, C2)
    for t, (u_c, v_c, w_c) in enumerate(terms):
        col = t + rank_half
        for (i, j, coeff) in u_c:
            U[i*m+j, col] += coeff
        for (j, k, coeff) in v_c:
            # B2 is B[:, 3:6], so k' = k+3
            V[j*p + (k+3), col] += coeff
        for (k, i, coeff) in w_c:
            # C2 is C[3:6, :], so k' = k+3
            W[(k+3)*n + i, col] += coeff

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
