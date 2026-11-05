import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The multiplication tensor T has components T_{i,j,k} = 1 if the indices match the multiplication pattern.
    # Specifically, for C = A * B, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T = sum_{i,j,k} e_{i,j}^{(A)} \otimes e_{j,k}^{(B)} \otimes e_{k,i}^{(C)}
    # Here A is n x m, B is m x p, C is p x n.
    # Indices:
    # A index: i (0..n-1), j (0..m-1) -> flat idx u = i*m + j
    # B index: j (0..m-1), k (0..p-1) -> flat idx v = j*p + k
    # C index: k (0..p-1), i (0..n-1) -> flat idx w = k*n + i  (Note: official trace convention W[k*n+i, r])
    
    # We want U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_{l=0}^{r-1} U[u, l] * V[v, l] * W[w, l] = T[u, v, w]
    # where T[u, v, w] = 1 if the indices (i,j) from u, (j',k) from v, (k',i') from w satisfy j=j' and i=i' and k=k'.
    # Wait, the tensor for matrix multiplication is:
    # T_{i,j,k} is not the right notation.
    # Let's use the standard definition:
    # M_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the flattened indices:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i  (as per task description: W[k*n+i, r])
    
    # So T[u, v, w] = 1 if:
    # i = u // m, j = u % m
    # j' = v // p, k = v % p
    # k' = w // n, i' = w % n
    # Condition: j == j' AND i == i' AND k == k'
    
    # We seek a decomposition of rank r.
    # This is equivalent to finding matrices U, V, W such that the tensor formed by sum_l U[:,l] \otimes V[:,l] \otimes W[:,l] equals T.
    
    # Strategy:
    # For a fixed rank r, this is a system of quadratic equations.
    # However, we can use a randomized linearization technique.
    # Fix U and V randomly (or structured), then solve for W? No, it's bilinear in W if U,V fixed?
    # sum_l U[u,l] V[v,l] W[w,l] = T[u,v,w]
    # This is linear in W for each fixed u,v? No, W[w,l] is multiplied by U[u,l]V[v,l].
    # If we fix U and V, we have a linear system for W?
    # Let M_{w, (u,v)} = sum_l U[u,l] V[v,l] \delta_{l, ?} -- No.
    
    # Let's rearrange:
    # For each w, we have:
    # sum_l ( sum_{u,v} T[u,v,w] U[u,l] V[v,l] ) ? No.
    
    # Let's look at the structure:
    # X_l = U[:,l] \otimes V[:,l] is a vector of length nm*mp.
    # The equation is: sum_l W[w,l] X_l[u,v] = T[u,v,w] for all u,v,w.
    # This is linear in W if we treat X_l as known coefficients.
    # But X_l depends on U and V.
    
    # Alternative Strategy: Randomized Linearization of the Bilinear Form.
    # We can choose U and V to be random matrices with entries from a small set (e.g., {0, 1}).
    # Then the system becomes linear in W?
    # Equation: sum_l U[u,l] V[v,l] W[w,l] = T[u,v,w]
    # This is NOT linear in W alone because the coefficient of W[w,l] is U[u,l]V[v,l], which depends on u,v.
    # So for a fixed w, we have a system of equations in variables W[w, 0..r-1].
    # The coefficient of W[w, l] in the equation for (u,v) is U[u,l] * V[v,l].
    # So for each w, we have a linear system:
    # A_w * W[w, :] = b_w
    # where A_w is a matrix of size (nm*mp) x r, with A_w[u,v, l] = U[u,l] * V[v,l].
    # and b_w is a vector of size (nm*mp) with b_w[u,v] = T[u,v,w].
    
    # If we fix U and V randomly, we can solve for W[w, :] for each w independently!
    # If a solution exists for all w, we have found a decomposition.
    # The entries of W will be rational numbers (if U,V are integers).
    # We need integer/half-integer entries.
    # If we choose U and V with entries in {0, 1}, the system A_w W_w = b_w might have rational solutions.
    # We can then check if the solutions are integers or half-integers.
    # If not, we can scale? No, scaling U,V,W changes the product.
    # However, if we find a solution with rational entries, we can potentially clear denominators by scaling U, V, W?
    # No, the rank is fixed. We cannot scale individual terms.
    # But we can choose U and V to have larger integer entries to increase the chance of integer solutions?
    # Or, we can just accept the rational solution if the evaluator allows it?
    # "integer/half-integer entries" is specified.
    
    # Let's try to find U, V such that the resulting W has integer entries.
    # This is a hard constraint.
    
    # Simpler Approach:
    # Use the known schoolbook decomposition (rank 100) as a baseline.
    # Then, try to reduce rank by solving the linear system for W given random U, V.
    # If we find a valid W with integer entries, we have a decomposition of rank r.
    
    # We will iterate r from 99 down to 50.
    # For each r, we try several random U, V.
    
    deadline = time.time() + 140
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank), dtype=int)
    best_V = np.zeros((mp, best_rank), dtype=int)
    best_W = np.zeros((pn, best_rank), dtype=int)
    
    # Initialize with schoolbook
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                best_U[u, idx] = 1
                best_V[v, idx] = 1
                best_W[w, idx] = 1
                idx += 1
                
    # Precompute T[u,v,w] as a 3D array or list of (u,v) pairs for each w
    # T is sparse. T[u,v,w] = 1 iff j(u)==j(v) and i(u)==i(w) and k(v)==k(w).
    # For a fixed w, the non-zero entries of b_w are at (u,v) satisfying the conditions.
    
    # Let's precompute the non-zero indices for each w.
    # w = k*n + i
    # i = w % n, k = w // n
    # Conditions:
    # i(u) = i
    # k(v) = k
    # j(u) = j(v)
    
    # For a fixed w (i,k):
    # u must be of form i*m + j
    # v must be of form j*p + k
    # So for each j in 0..m-1:
    # u = i*m + j
    # v = j*p + k
    # T[u,v,w] = 1
    
    # So for each w, b_w has exactly m non-zero entries (one for each j).
    # The system A_w W_w = b_w has nm*mp equations, but most b_w entries are 0.
    # The equations with b_w[u,v] = 0 must be satisfied as well:
    # sum_l U[u,l] V[v,l] W[w,l] = 0
    
    # This is a homogeneous system for the zero entries, and inhomogeneous for the non-zero ones.
    # We can solve the full system using least squares or exact linear solve if square/overdetermined.
    # Since r < nm*mp, the system is overdetermined.
    # We need an exact solution.
    # We can use numpy.linalg.lstsq and check the residual.
    # If residual is 0 (within tolerance) and entries are integer/half-integer, we have a solution.
    
    # To improve chances of integer solutions, we can scale U and V?
    # No, let's just try random U, V with small integers.
    
    for r in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random U, V
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                # Build A_w and b_w
                # A_w is (nm*mp) x r
                # A_w[u,v, l] = U[u,l] * V[v,l]
                # We can compute this efficiently.
                # A_w = (U.T @ V.T).T ? No.
                # A_w[u,v,l] = U[u,l] * V[v,l]
                # This is the Khatri-Rao product of U and V, reshaped.
                # Let's compute it.
                
                # For efficiency, we can solve the system only for the non-zero b_w entries?
                # No, the zero entries provide constraints too.
                # But if we find a solution that satisfies the non-zero entries, it might not satisfy the zero ones.
                # We must check all constraints.
                
                # Let's build the full system.
                # A_w = np.einsum('ul,vl->uvl', U, V).reshape(nm*mp, r)
                # This is expensive for large r, but r is small (<100).
                # nm*mp = 20*25 = 500.
                # 500 x r matrix.
                
                # We can avoid building the full matrix if we use a solver that handles sparse?
                # But the structure is dense in A_w.
                
                # Let's try to solve A_w W_w = b_w
                # b_w is sparse.
                
                # Optimization:
                # Instead of solving the full 500x r system, we can pick a subset of rows?
                # No, we need exactness.
                
                # Let's just build A_w.
                A_w = np.einsum('ul,vl->uvl', U, V).reshape(nm * mp, r)
                
                # b_w
                b_w = np.zeros(nm * mp)
                i_w = w % n
                k_w = w // n
                for j in range(m):
                    u = i_w * m + j
                    v = j * p + k_w
                    b_w[u * mp + v] = 1
                
                # Solve A_w W_w = b_w
                # Use lstsq
                try:
                    W_w, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check residual
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        valid = False
                        break
                else:
                    # If rank < r, lstsq returns min-norm solution.
                    # Check residual directly
                    res = np.linalg.norm(A_w @ W_w - b_w)
                    if res > 1e-9:
                        valid = False
                        break
                
                # Check if W_w entries are integer or half-integer
                # Multiply by 2 to check for integers
                W_w_scaled = W_w * 2
                if not np.allclose(W_w_scaled, np.round(W_w_scaled), atol=1e-6):
                    valid = False
                    break
                
                W[w] = W_w
            
            if valid:
                # We have a valid decomposition with integer/half-integer entries
                # Convert W to integer if possible? The task says integer/half-integer.
                # If W_w_scaled is integer, W_w is half-integer.
                # We can return W as is (floats) or convert to a format that preserves exactness.
                # The evaluator likely checks the product.
                # If we return floats, they might have precision issues.
                # Better to return the scaled integers? No, the interface expects the actual values.
                # We can return W as a list of floats.
                
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break # Found a lower rank, stop searching for this r
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
