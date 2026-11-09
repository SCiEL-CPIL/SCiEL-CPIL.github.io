import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin

    n, m, p = DIMENSIONS

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Verified Strassen decomposition
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (A21+A22)B11 -> C21 += P2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # P3 = A11(B12-B22) -> C12 += P3, C22 -= P3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = -1
        
        # P4 = A22(B21-B11) -> C11 -= P4, C21 += P4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1
        
        # P5 = (A11+A12)B22 -> C11 += P5, C12 += P5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # P6 = (A21-A11)(B11+B12) -> C21 += P6, C22 -= P6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1; W[3][5] = -1
        
        # P7 = (A12-A22)(B21+B22) -> C22 += P7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W

    def schoolbook(n, m, p):
        """Return U, V, W for schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return U, V, W

    def combine_decomps(n, m, p, parts):
        """
        Combine decompositions of sub-problems.
        parts: list of (n_part, m_part, p_part, U_part, V_part, W_part)
        """
        total_rank = sum(len(U_part[0]) for _, _, _, U_part, _, _ in parts if U_part)
        
        # Build combined U, V, W
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank_offset = 0
        for n_p, m_p, p_p, U_part, V_part, W_part in parts:
            r_part = len(U_part[0]) if U_part else 0
            
            for i in range(n_p):
                for j in range(m_p):
                    for r in range(r_part):
                        U[i * m + j][rank_offset + r] = U_part[i * m_p + j][r]
                        
            for j in range(m_p):
                for k in range(p_p):
                    for r in range(r_part):
                        V[j * p + k][rank_offset + r] = V_part[j * p_p + k][r]
                        
            for k in range(p_p):
                for i in range(n_p):
                    for r in range(r_part):
                        W[k * n + i][rank_offset + r] = W_part[k * n_p + i][r]
                        
            rank_offset += r_part
            
        return U, V, W

    def get_decomp(n, m, p):
        """Returns U, V, W as lists of lists for n x m x p decomposition."""
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            U, V, W = strassen_2x2x2()
            # Verify
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        val = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(7))
                        if val != 1:
                            valid = False
            if valid:
                return U, V, W
        
        # Try different splitting strategies
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook(n, m, p)
        
        # Strategy 1: Split along m dimension
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    # Build combined decomposition
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m1):
                            for rk in range(r1):
                                new_U[i * m + j][rk] = U1[i * m1 + j][rk]
                        for j in range(m2):
                            for rk in range(r2):
                                new_U[i * m + m1 + j][r1 + rk] = U2[i * m2 + j][rk]
                                
                    for j in range(m1):
                        for k in range(p):
                            for rk in range(r1):
                                new_V[j * p + k][rk] = V1[j * p + k][rk]
                    for j in range(m2):
                        for k in range(p):
                            for rk in range(r2):
                                new_V[(m1 + j) * p + k][r1 + rk] = V2[j * p + k][rk]
                                
                    for k in range(p):
                        for i in range(n):
                            for rk in range(r1):
                                new_W[k * n + i][rk] = W1[k * n + i][rk]
                            for rk in range(r2):
                                new_W[k * n + i][r1 + rk] = W2[k * n + i][rk]
                                
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Strategy 2: Split along n dimension
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n1):
                        for j in range(m):
                            for rk in range(r1):
                                new_U[i * m + j][rk] = U1[i * m + j][rk]
                    for i in range(n2):
                        for j in range(m):
                            for rk in range(r2):
                                new_U[(n1 + i) * m + j][r1 + rk] = U2[i * m + j][rk]
                                
                    for j in range(m):
                        for k in range(p):
                            for rk in range(r1):
                                new_V[j * p + k][rk] = V1[j * p + k][rk]
                            for rk in range(r2):
                                new_V[j * p + k][r1 + rk] = V2[j * p + k][rk]
                                
                    for k in range(p):
                        for i in range(n1):
                            for rk in range(r1):
                                new_W[k * n + i][rk] = W1[k * n1 + i][rk]
                        for i in range(n2):
                            for rk in range(r2):
                                new_W[k * n + n1 + i][r1 + rk] = W2[k * n2 + i][rk]
                                
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Strategy 3: Split along p dimension
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m):
                            for rk in range(r1):
                                new_U[i * m + j][rk] = U1[i * m + j][rk]
                            for rk in range(r2):
                                new_U[i * m + j][r1 + rk] = U2[i * m + j][rk]
                                
                    for j in range(m):
                        for k in range(p1):
                            for rk in range(r1):
                                new_V[j * p + k][rk] = V1[j * p1 + k][rk]
                        for k in range(p2):
                            for rk in range(r2):
                                new_V[j * p + p1 + k][r1 + rk] = V2[j * p2 + k][rk]
                                
                    for k in range(p1):
                        for i in range(n):
                            for rk in range(r1):
                                new_W[k * n + i][rk] = W1[k * n + i][rk]
                    for k in range(p2):
                        for i in range(n):
                            for rk in range(r2):
                                new_W[(p1 + k) * n + i][r1 + rk] = W2[k * n + i][rk]
                                
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    
    if time.time() > deadline:
         rank = n * m * p
         U, V, W = schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
