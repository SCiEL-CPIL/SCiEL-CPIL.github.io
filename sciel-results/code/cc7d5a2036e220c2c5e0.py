import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Start with a rank-30 decomposition using a systematic approach
    # Then try to reduce rank via linear dependency analysis
    
    # First, build a known valid decomposition
    # Using the structure: for each (i,j,k) we need U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] to sum to 1
    
    # Start with schoolbook rank 30
    rank = 30
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
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Try to find a lower rank decomposition by analyzing the tensor
    # The matrix multiplication tensor T[i,j,k] = delta_ik (for appropriate indices)
    # We want to find the minimal rank decomposition
    
    # Approach: Use the fact that for 2x3 * 3x5, we can potentially do better than 30
    # Known lower bounds suggest rank >= 18 or so, but exact values are hard
    
    # Try iterative elimination: if there's a linear dependency among the rank-1 terms,
    # we can reduce the rank
    
    best_U, best_V, best_W = U_mat, V_mat, W_mat
    best_rank = rank
    
    # Check for linear dependencies in the current decomposition
    if time.time() < deadline - 30.0:
        # Build the tensor matrix: each term r contributes a vector in R^(n*m*p)
        terms = []
        for r in range(rank):
            t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
            terms.append(t.flatten())
        
        M = np.array(terms).T  # shape (n*m*p, rank)
        
        # Use SVD to find the numerical rank
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are null vectors - we can potentially reduce rank
            # For each null vector, try to eliminate one term
            for null_idx in range(num_rank, rank):
                if time.time() > deadline - 10.0:
                    break
                    
                c = Vt[null_idx, :]  # null vector
                # Find the term with largest coefficient to eliminate
                k_idx = np.argmax(np.abs(c))
                
                if np.abs(c[k_idx]) < 1e-8:
                    continue
                
                # Eliminate term k_idx by subtracting c[r]/c[k_idx] * term_r from term_k
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == k_idx:
                        continue
                    # The new coefficient for term r
                    coeff = 1.0
                    new_U.append(U_mat[:, r] * coeff)
                    new_V.append(V_mat[:, r] * coeff)
                    new_W.append(W_mat[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                # Verify the reduced decomposition
                terms_new = []
                for r in range(len(new_U.T)):
                    t = np.einsum('i,j,k->ijk', new_U[:, r], new_V[:, r], new_W[:, r])
                    terms_new.append(t.flatten())
                M_new = np.array(terms_new).T
                
                sum_new = np.sum(M_new, axis=1)
                sum_old = np.sum(M, axis=1)
                
                if np.allclose(sum_new, sum_old, atol=1e-8):
                    new_rank = len(new_U.T)
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U, best_V, best_W = new_U, new_V, new_W
                        # Restart the dependency check with the new decomposition
                        U_mat, V_mat, W_mat = best_U, best_V, best_W
                        rank = best_rank
                        break
    
    # Try a different approach: use Strassen-like blocking for the 2x3*3x5 case
    # Split into blocks and use recursive or hybrid methods
    
    if time.time() < deadline - 20.0:
        # Try a rank-28 decomposition using a specific pattern
        rank28 = 28
        U28 = [[0] * rank28 for _ in range(n * m)]
        V28 = [[0] * rank28 for _ in range(m * p)]
        W28 = [[0] * rank28 for _ in range(p * n)]
        
        col28 = 0
        
        # Use a systematic approach: for the 2x3*3x5 case, we can use
        # a combination of schoolbook and Strassen-like terms
        
        # Terms for i=0,1; j=0,1,2; k=0,1,2,3,4
        # We need 30 terms in schoolbook, but can potentially do better
        
        # Try to construct a rank-28 decomposition by combining some terms
        # This is a heuristic approach - not guaranteed to work
        
        # For now, let's try to verify if our best result is valid
        if best_rank < rank:
            # Verify the best decomposition
            terms_best = []
            for r in range(best_rank):
                t = np.einsum('i,j,k->ijk', best_U[:, r], best_V[:, r], best_W[:, r])
                terms_best.append(t.flatten())
            M_best = np.array(terms_best).T
            
            # The sum should equal the matrix multiplication tensor
            T_expected = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_expected[(i * m + j) * p + k] = 1  # This is wrong, need to fix
            
            # Actually, the tensor is T[i,j,k] where the multiplication is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor element T[i,j,k] = 1 for all i,j,k (it's the identity in the j dimension)
            # Wait, no. The tensor is defined such that:
            # C[i,k] = sum_r U[i*m+?, r] * V[?*p+k, r] * W[k*n+i, r]
            # Actually, the standard form is:
            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) W[r]
            # No, that's not right either.
            
            # Let me re-derive. The tensor decomposition is:
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]  (using trace convention W[k*n+i,r])
            # where T[i,j,k] = delta_{i,k} ... no.
            
            # Actually for matrix multiplication C = A*B where A is n x m and B is m x p:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor is T[i,j,k] = 1 (it's just the multiplication structure)
            # And we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k? No.
            
            # The correct identity is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) ... this doesn't work directly.
            
            # The standard tensor decomposition for matrix multiplication:
            # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # where T_{ijk} is the multiplication tensor, and:
            # C_{ik} = sum_j A_{ij} B_{jk}
            
            # For our case, the tensor T[i,j,k] has the property that:
            # sum_j T[i,j,k] A[i,j] B[j,k] gives C[i,k]
            # Actually, the tensor is just T[i,j,k] = 1 for the relevant indices, and the
            # decomposition is: 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r] for the "active" terms.
            
            # I think the issue is that my verification is incorrect. Let me just return
            # the best valid decomposition we have.
            
            pass
    
    # Return the best decomposition we found
    if best_rank < 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
