"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try block-based Strassen approach
    # Partition: n=4=2+2, m=5=3+2, p=6=3+3
    # 2x2x2 blocks: 8 blocks
    # For 2x2x2: Strassen-2 rank 7
    # For 2x3x3: schoolbook rank 18
    # For 3x2x3: schoolbook rank 18
    # For 3x3x3: Strassen-3 rank 7
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def strassen3_decomp():
        """3x3x3 schoolbook, rank 27"""
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W
    
    # Try partition n=2+2, m=3+2, p=3+3
    # Blocks: (2,3,3),(2,3,3),(2,2,3),(2,2,3),(2,3,3),(2,3,3),(2,2,3),(2,2,3)
    # Ranks: 18,18,12,12,18,18,12,12 = total 120. Not better.
    
    # Search for rank-119 decomposition by introducing a shared parameter
    # that creates linear dependencies across blocks
    
    def try_rank_119(param):
        """Try to construct a rank-119 decomposition with given parameter."""
        # Use partition n=3+1, m=3+2, p=3+3
        # But modify one block to share a factor with another
        
        # Strategy: In block (0,0,0) [3x3x3], modify one rank-1 term
        # to be scaled by 'param', and in block (0,1,0) [3x2x3], 
        # add a compensating term
        
        total_rank = 119
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        # Block (0,0,0): n=3,m=3,p=3 -> schoolbook with one modified term
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    scale = param if (i == 0 and j == 0 and k == 0) else 1
                    U_full[i*m+j][col] = scale
                    V_full[j*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 27
        
        # Block (0,0,1): n=3,m=3,p=3 -> schoolbook
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 54
        
        # Block (0,1,0): n=3,m=2,p=3 -> schoolbook, but we need to compensate
        # for the modification in block (0,0,0)
        # The (0,0,0) term in block (0,0,0) contributes param * 1 * 1 to C[0,0]
        # We need total contribution = 1, so we need (1-param) from elsewhere
        # Add a compensating term using indices from block (0,1,0)
        # But block (0,1,0) uses j=3,4 (m indices), not j=0
        # So we can't directly compensate. Let's try a different approach.
        
        # Alternative: Modify the structure to allow sharing
        # Use rank-119 with one column being a linear combination
        
        # Reset and try a cleaner approach
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        # Block (0,0,0): n=3,m=3,p=3 -> 27 terms, but use param scaling
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 27
        
        # Block (0,0,1): n=3,m=3,p=3 -> 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 54
        
        # Block (0,1,0): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 72
        
        # Block (0,1,1): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 90
        
        # Block (1,0,0): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 99
        
        # Block (1,0,1): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 108
        
        # Block (1,1,0): n=1,m=2,p=3 -> 6 terms
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 114
        
        # Block (1,1,1): n=1,m=2,p=3 -> 5 terms (one less than schoolbook)
        # We save one rank by making one column a linear combination
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    # Skip the last term (k=2, j=1, i=0)
                    if j == 1 and k == 2:
                        continue
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 119
        
        # Now verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        return None
    
    # Try a few parameter values (though the above doesn't use param yet)
    # Let's try a smarter approach: use a shared column
    
    def try_shared_column():
        """Try rank-119 by sharing one column between blocks."""
        total_rank = 119
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        # Block (0,0,0): n=3,m=3,p=3 -> 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 27
        
        # Block (0,0,1): n=3,m=3,p=3 -> 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 54
        
        # Block (0,1,0): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 72
        
        # Block (0,1,1): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 90
        
        # Block (1,0,0): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 99
        
        # Block (1,0,1): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 108
        
        # Block (1,1,0): n=1,m=2,p=3 -> 6 terms
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 114
        
        # Block (1,1,1): n=1,m=2,p=3 -> 5 terms
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    if j == 1 and k == 2:
                        continue
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 119
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        return None
    
    # Try the shared column approach
    result = try_shared_column()
    if result is not None:
        return result
    
    # If that fails, try a randomized search with small perturbations
    import random
    random.seed(42)
    
    def try_random_perturbation():
        """Try rank-119 with random small perturbations to create dependencies."""
        total_rank = 119
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        # Block (0,0,0): n=3,m=3,p=3 -> 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 27
        
        # Block (0,0,1): n=3,m=3,p=3 -> 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 54
        
        # Block (0,1,0): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
        # col = 72
        
        # Block (0,1,1): n=3,m=2,p=3 -> 18 terms
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U_full[i*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+i][col] = 1
                    col += 1
        # col = 90
        
        # Block (1,0,0): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 99
        
        # Block (1,0,1): n=1,m=3,p=3 -> 9 terms
        for i in range(1):
            for j in range(3):
                for k in range(3):
                    U_full[(i+3)*m+j][col] = 1
                    V_full[j*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 108
        
        # Block (1,1,0): n=1,m=2,p=3 -> 6 terms
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+k][col] = 1
                    W_full[k*n+(i+3)][col] = 1
                    col += 1
        # col = 114
        
        # Block (1,1,1): n=1,m=2,p=3 -> 5 terms with one perturbation
        # Use a perturbation that might create a dependency
        eps = random.choice([0.5, -0.5, 0.25, -0.25, 0.75, -0.75])
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    if j == 1 and k == 2:
                        continue
                    U_full[(i+3)*m+(j+3)][col] = 1
                    V_full[(j+3)*p+(k+3)][col] = 1
                    W_full[(k+3)*n+(i+3)][col] = 1
                    col += 1
        # col = 119
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        return None
    
    # Try a few random perturbations
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        result = try_random_perturbation()
        if result is not None:
            return result
    
    # Return the schoolbook solution as fallback
    return best_result
# EVOLVE-BLOCK-END
