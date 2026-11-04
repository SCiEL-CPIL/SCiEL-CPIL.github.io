"""Find rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor: T[(i,j), (j,k), (k,i)] = 1 for all valid triples
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Precompute all (i,j,k) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for (u_idx, v_idx, w_idx) in triples:
            s = 0.0
            for c in range(rank):
                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2
    
    def is_half_int(x, tol=1e-9):
        return abs(x * 2 - round(x * 2)) < tol
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 7)
    # Embed 2x3 and 3x5 into 3x3 with zero-padding:
    # A_3x3 = [[a00, a01, a02], [a10, a11, a12], [0, 0, 0]]
    # B_3x3 = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14], [b20, b21, b22, b23, b24], [0,0,0,0,0], [0,0,0,0,0]]
    # Wait, B is 3x5, so embedded in 3x3 doesn't work directly.
    # 
    # Let me reconsider. We need to multiply (2x3) * (3x5) = 2x5.
    # Pad A to 3x3: A' = [[a00,a01,a02],[a10,a11,a12],[0,0,0]]
    # Pad B to 3x3: B' = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # But B is 3x5, not 3x3. We can't pad a 3x5 into 3x3.
    #
    # Alternative: Pad both to larger square matrices.
    # Pad A (2x3) to 5x5: A' with rows 0,1 having the 3 elements + 2 zeros, rows 2-4 all zeros
    # Pad B (3x5) to 5x5: B' with cols 0-4 having the 3 rows + 2 zero rows
    # Then A'*B' is 5x5, and the top-left 2x5 block is our answer.
    # But 5x5 multiplication has rank at least 25, which is worse.
    #
    # Better approach: Use the structure directly.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3,4
    # 
    # Let's use a block decomposition:
    # Split columns of B into B1 (3x2) and B2 (3x3): B = [B1 | B2]
    # Then C = [A*B1 | A*B2]
    # A*B1: 2x2 multiplication, can use Strassen (rank 7)
    # A*B2: 2x3 * 3x3 multiplication
    #
    # For 2x3 * 3x3: pad A to 3x3 (add zero row), pad result to 3x3
    # Use Laderman (rank 7) for 3x3*3x3, but we only need top 2 rows of result
    # The zero row in A means some Laderman terms vanish.
    #
    # Let me try a different approach: direct algebraic decomposition.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using Winograd-style: for 2x3 * 3x5, we can try to find a low-rank decomposition.
    # The schoolbook rank is 2*3*5 = 30.
    # 
    # Let's try: split into (2x2)*(2x5) + (2x1)*(1x5) or similar.
    # Actually, let's use the known result that 2x3*3x3 can be done in rank 16
    # (Strassen 2x2 rank 7 + additional terms).
    
    # Let me try a systematic approach: use the structure to find a decomposition.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Try to factor this. One approach:
    # Let's use the identity: x*y = ((x+y)^2 - x^2 - y^2)/2 (not directly helpful)
    #
    # For 2x3 * 3x5, let me try to use a Kronecker product approach:
    # (2x3)*(3x5) = (2x2)*(2x5) + (2x1)*(1x5)
    # where we split the 3 columns into 2+1.
    # 
    # (2x2)*(2x5): rank at most 7*5 = 35 (Strassen per column), but we can do better.
    # Actually, (2x2)*(2x5) can be done as 5 separate (2x2)*(2x1) multiplications.
    # Each (2x2)*(2x1) has rank 4 (schoolbook) or can be done in rank 4.
    # Total: 5*4 = 20 for this part.
    # (2x1)*(1x5): rank 2*1*5 = 10 (schoolbook) or can be optimized.
    # Total: 20 + 10 = 30. No improvement.
    
    # Let me try another split: (2x3)*(3x5) = (2x2)*(2x5) + (2x1)*(1x5)
    # But do (2x2)*(2x5) using Strassen-like: 
    # (2x2)*(2x5) = (2x2)*(2x2) + (2x2)*(2x3)
    # This is getting complicated. Let me just use the search approach but with a better initialization.
    
    # Actually, let me try the Laderman embedding more carefully.
    # Embed 2x3 into 3x3: A3 = [[a00,a01,a02],[a10,a11,a12],[0,0,0]]
    # Embed 3x5 into 5x5: B5 = [[b00,b01,b02,b03,b04],[b10,b11,b12,b13,b14],[b20,b21,b22,b23,b24],[0,0,0,0,0],[0,0,0,0,0]]
    # Pad A3 to 5x5: A5 = [[a00,a01,a02,0,0],[a10,a11,a12,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    # A5*B5 is 5x5. Top-left 2x5 is our answer.
    # But 5x5*5x5 has rank >= 25.
    
    # Let me try a completely different approach: use the fact that we only need 10 outputs.
    # C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    # 
    # Try to find a rank-r decomposition where r < 30.
    # Use the algebraic structure: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # One known approach: use the "border rank" or "border decomposition".
    # For now, let's use a smarter search.
    
    def try_rank(rank, max_attempts=100):
        """Try to find a rank-r decomposition using random initialization with gradient descent."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                return None
            
            np.random.seed(attempt)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Gradient descent
            lr = 0.05
            for step in range(500):
                if time.time() > deadline - 30:
                    break
                
                # Compute error and gradients
                error = 0.0
                gU = np.zeros((nm, rank))
                gV = np.zeros((mp, rank))
                gW = np.zeros((pn, rank))
                
                for (u_idx, v_idx, w_idx) in triples:
                    s = 0.0
                    for c in range(rank):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    err = s - 1.0
                    error += err * err
                    for c in range(rank):
                        gU[u_idx, c] += 2 * err * V[v_idx, c] * W[w_idx, c]
                        gV[v_idx, c] += 2 * err * U[u_idx, c] * W[w_idx, c]
                        gW[w_idx, c] += 2 * err * U[u_idx, c] * V[v_idx, c]
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-12:
                    break
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Decay learning rate
                lr *= 0.995
            
            # Try to round to half-integers
            if best_error < 1e-6:
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                valid = True
                for (u_idx, v_idx, w_idx) in triples:
                    s = 0.0
                    for c in range(rank):
                        s += U_rounded[u_idx, c] * V_rounded[v_idx, c] * W_rounded[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    U_list = [[float(U_rounded[i][c]) for c in range(rank)] for i in range(nm)]
                    V_list = [[float(V_rounded[i][c]) for c in range(rank)] for i in range(mp)]
                    W_list = [[float(W_rounded[i][c]) for c in range(rank)] for i in range(pn)]
                    return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from low to high
    for rank in range(16, 31):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
