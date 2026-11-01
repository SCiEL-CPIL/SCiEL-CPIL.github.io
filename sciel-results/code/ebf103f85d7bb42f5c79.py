"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p  # 60 equations
    
    # Target tensor: all ones (each entry is 1)
    # We need: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    
    # Try ranks from a lower bound upward
    # Lower bound: at least max(n, p) * m? No. Let's just try.
    # Known: for 2x2x2, rank is 7. For larger, it's complex.
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Try various ranks
    for rank in range(1, 60):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Use iterative refinement: treat as a system and use least squares / gradient
            # The equations are: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a polynomial system. Let's use a simple approach:
            # Fix two matrices, solve for the third, iterate.
            
            for it in range(500):
                if time.time() > deadline - 3:
                    break
                    
                # Solve for W given U, V:
                # For each (k,i): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # This is: for each (k,i), sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Wait, that's not right. The equation is for each (i,j,k) separately.
                
                # Let me re-express. For fixed i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # For fixed i,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                
                # Actually, let's just do gradient descent on the full system.
                # Compute residual and gradient.
                
                # Build residual: shape (n, m, p)
                # R[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
                # This is expensive to compute naively. Let's use matrix operations.
                
                # Alternative: use the fact that for each c, the contribution is outer product
                # U_c (length nm) * V_c (length mp) * W_c (length pn)
                # The (i,j,k) entry is U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
                
                # Let's compute residual using einsum-like approach
                # R = zeros((n,m,p))
                # For each c: R += np.einsum('a,b,c->ijk', 
                #     U[:,c].reshape(n,m), V[:,c].reshape(m,p), W[:,c].reshape(p,n))
                # But W is indexed by [k*n+i], so W_c[k*n+i] for k in 0..p-1, i in 0..n-1
                # So W_c.reshape(p,n)[k,i]
                
                R = np.zeros((n, m, p))
                for c in range(rank):
                    uc = U[:, c].reshape(n, m)
                    vc = V[:, c].reshape(m, p)
                    wc = W[:, c].reshape(p, n)
                    # Contribution: uc[i,j] * vc[j,k] * wc[k,i]
                    # This is a tensor product
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
                R -= 1.0
                
                # Check convergence
                if np.max(np.abs(R)) < 1e-8:
                    found = True
                    break
                
                # Compute gradient
                # dR/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] summed over k... 
                # No, for each (i,j,k), dR[i,j,k]/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                # So the gradient for U[i*m+j,c] is sum_k R[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rijk = R[i, j, k]
                            for c in range(rank):
                                dU[i*m+j, c] += rijk * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += rijk * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += rijk * U[i*m+j, c] * V[j*p+k, c]
                
                # Gradient descent with adaptive step
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                # Verify and convert to list
                # Check all entries
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1.0) > 1e-6:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    # Convert to lists with rounding to half-integers if possible
                    U_list = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(nm)]
                    V_list = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(mp)]
                    W_list = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(pn)]
                    
                    # Verify with rounded values
                    ok2 = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(rank):
                                    val += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                                if abs(val - 1.0) > 1e-6:
                                    ok2 = False
                                    break
                        if not ok2:
                            break
                    
                    if ok2:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
                    else:
                        # Use unrounded values
                        U_list = [[U[i, c] for c in range(rank)] for i in range(nm)]
                        V_list = [[V[i, c] for c in range(rank)] for i in range(mp)]
                        W_list = [[W[i, c] for c in range(rank)] for i in range(pn)]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
    
    # Fallback: schoolbook
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
