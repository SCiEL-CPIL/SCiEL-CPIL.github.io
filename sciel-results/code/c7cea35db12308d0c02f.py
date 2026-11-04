"""Systematic rank-1 peeling algorithm for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 30
    
    # We want to find a rank-minimal decomposition.
    # The tensor T has entries T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to the (i,k) entry of C. In our indexing:
    # U has shape (n*m, rank), indexed by (i, j)
    # V has shape (m*p, rank), indexed by (j, k)
    # W has shape (p*n, rank), indexed by (k, i)
    # The condition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,i} * delta_{j,j} * delta_{k,k}... 
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor of matmul)
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try to find a decomposition of rank r < 30.
    # We'll use a greedy approach: try to peel off rank-1 terms.
    
    # First, let's set up the tensor as a 3D array
    # T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    T = np.ones((n, m, p))
    
    # We'll try ranks from a lower bound up to the schoolbook rank
    # Lower bound for 2x3x5: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Actually, the trivial lower bound is max(n*m, m*p, p*n) but there might be better bounds.
    # For 2x3x5, let's try from 15 up to 29.
    
    for target_rank in range(15, rank_schoolbook):
        result = _try_decomposition(n, m, p, target_rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_decomposition(n, m, p, target_rank):
    """Try to find a rank-target_rank decomposition using rank-1 peeling."""
    # Start with the full tensor
    residual = np.ones((n, m, p))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(target_rank):
        # Try to find a rank-1 term u ⊗ v ⊗ w such that
        # residual - alpha * (u ⊗ v ⊗ w) has integer/half-integer entries
        # and the remaining residual can be completed with (target_rank - step - 1) more terms.
        
        # For simplicity, we'll try to find u, v, w such that
        # residual[i,j,k] - u[i]*v[j]*w[k] is "nice" (integer or half-integer)
        # and the new residual is still "decomposable".
        
        # A simpler approach: try all possible rank-1 tensors with small integer/half-integer
        # entries and see if the residual remains decomposable.
        
        # Actually, let's use a different strategy: 
        # We know that for the matrix multiplication tensor, each entry is 1.
        # A rank-1 term contributes u[i]*v[j]*w[k] to entry (i,j,k).
        
        # Let's try to find u, v, w in {0, 1/2, 1} such that the residual
        # after subtracting u⊗v⊗w is still "close" to a sum of rank-1 tensors.
        
        # For a more systematic approach, let's try all combinations of u, v, w
        # with entries in {-1, -1/2, 0, 1/2, 1} and check if the residual
        # can be decomposed with the remaining rank.
        
        found = False
        candidates = [-1, -0.5, 0, 0.5, 1]
        
        for u_vals in product(candidates, repeat=n):
            u = np.array(u_vals)
            if np.all(u == 0):
                continue
            for v_vals in product(candidates, repeat=m):
                v = np.array(v_vals)
                if np.all(v == 0):
                    continue
                for w_vals in product(candidates, repeat=p):
                    w = np.array(w_vals)
                    if np.all(w == 0):
                        continue
                    
                    rank1 = np.einsum('i,j,k->ijk', u, v, w)
                    new_residual = residual - rank1
                    
                    # Check if new_residual has entries that are "nice"
                    # (integer or half-integer)
                    if not _is_nice(new_residual):
                        continue
                    
                    # Check if new_residual can be decomposed with (target_rank - step - 1) more terms
                    # For now, just check if it's "small enough" or zero
                    if np.allclose(new_residual, 0, atol=1e-10):
                        U_cols.append(u)
                        V_cols.append(v)
                        W_cols.append(w)
                        residual = np.zeros((n, m, p))
                        found = True
                        break
                    elif step == target_rank - 1:
                        # This is the last term, so new_residual must be zero
                        continue
                    else:
                        # Heuristic: check if the new residual has a "simple" structure
                        # that might be decomposable
                        if _is_decomposable_heuristic(new_residual, target_rank - step - 1):
                            U_cols.append(u)
                            V_cols.append(v)
                            W_cols.append(w)
                            residual = new_residual
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            return None
    
    if np.allclose(residual, 0, atol=1e-10):
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return U, V, W
    return None


def _is_nice(arr, tol=1e-10):
    """Check if all entries are integer or half-integer."""
    for val in arr.flat:
        # Check if val is close to an integer or half-integer
        nearest_int = round(val)
        nearest_half = round(2 * val) / 2
        if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
            return False
    return True


def _is_decomposable_heuristic(residual, remaining_rank):
    """Heuristic check if residual can be decomposed with remaining_rank terms."""
    # If residual is zero, yes
    if np.allclose(residual, 0, atol=1e-10):
        return True
    
    # Check if the residual has a "simple" structure
    # For now, just check if all entries are small
    if np.max(np.abs(residual)) > 2:
        return False
    
    # Check if the residual is "rank-deficient" in some sense
    # This is a weak heuristic
    return True


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1) > 1e-10:
                    return False
    return True
