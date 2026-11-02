"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
DEADLINE_MARGIN = 10  # seconds before kill

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check tensor decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), the bilinear form must match
    # Actually the tensor is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # And W should satisfy: W[k*n+i, c] is the coefficient
    # Wait, let me re-read the convention.
    # The decomposition means: (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # No wait, standard tensor decomposition:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}? 
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # The matrix multiplication tensor T has entries T[i,j,k] = delta (i,j,k form a valid multiplication)
    # Actually for matrix mult: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor is T[i,j,k] which when contracted gives (AB)_{ik}
    # Tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But that's a 3-tensor decomposition with 3 factors.
    # Wait, the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is indexed by (i,j), V by (j,k), W by (k,i)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, that doesn't make sense dimensionally for matrix multiplication.
    # 
    # Let me reconsider. The matrix multiplication tensor for multiplying (n x m) by (m x p) to get (n x p):
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # This is a bilinear map. The "tensor" here is the structure tensor.
    # A rank-r decomposition means we can write:
    # (AB)_{ik} = sum_{r=1}^R (sum_j U[i*m+j, r] * A_{ij}) * (sum_j V[j*p+k, r] * B_{jk})
    # where the W factor encodes how to combine: W[k*n+i, r] is the coefficient.
    # Actually no. Let me think again.
    #
    # The standard formulation: matrix multiplication is a trilinear form or bilinear map.
    # The tensor T has T[i,j,k] = 1 if these indices form a valid term in the product.
    # For (AB)_{ik} = sum_j A_{ij}B_{jk}, the tensor is T[i,j,k] which is 1 for all valid (i,j,k).
    # 
    # A rank-r decomposition of this tensor: T[i,j,k] = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # But that's not right either because W would be redundant.
    #
    # Actually, I think the convention is:
    # (AB)_{ik} = sum_r (sum_j U[i*m+j, r] A_{ij}) * (sum_j V[j*p+k, r] B_{jk})
    # And W[k*n+i, r] must equal 1 for the decomposition to be valid? No...
    #
    # Let me look at the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So (AB)_{ik} = sum_{j} A_{ij} B_{jk} = sum_col (sum_j U[i*m+j,col] A_{ij}) * (sum_j V[j*p+k,col] B_{jk})
    # In schoolbook, each col corresponds to one (i,j,k) triple, so:
    # sum_col (sum_j U[i*m+j,col] A_{ij}) * (sum_j V[j*p+k,col] B_{jk})
    # = sum_col A[i,j_col] * B[j_col, k]  (since U and V pick out specific j)
    # = sum_j A[i,j] B[j,k]  (summing over all j via different cols)
    # So W is not directly used in the product formula? 
    #
    # I think the correct interpretation is:
    # The tensor T[i,j,k] for matrix mult is defined such that the bilinear form is:
    # <T, A, B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # And T[i,j,k] = 1 for all valid indices (it's just the identity structure).
    # 
    # A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this is a 3-way decomposition which is unusual.
    #
    # Actually, I think I'm overcomplicating. Let me just verify the schoolbook works:
    # The claim is that these three matrices "exactly decompose the matrix multiplication tensor."
    # The verification should check that the tensor identity holds.
    #
    # For the bilinear matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # With the decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 (since all (i,j,k) triples contribute).
    #
    # Wait no, that's not right. T[i,j,k] should be 1 for all i,j,k in valid range.
    # In schoolbook: for each (i,j,k), there's one column where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # So sum_r ... = 1 for each (i,j,k). Yes! That works.
    
    # So verify: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, time_limit=60):
    """Try to find a decomposition of given rank. Returns dict or None."""
    start = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    total_constraints = n * m * p  # one per (i,j,k)
    
    # We need: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of trilinear equations.
    # 
    # Strategy: Fix U and V, solve for W (linear in W).
    # For each (i,j,k): W[k*n+i, r] appears in the equation for (i,j,k).
    # But W[k*n+i, r] appears in multiple equations (one for each j).
    # So for fixed U, V: for each (i,k), the equations over j give:
    # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = n... wait no, = 1 for each j separately.
    # 
    # Actually each (i,j,k) gives a separate equation. W[k*n+i,r] appears in equations for all j.
    # So for fixed U,V, we have a linear system in W.
    #
    # Let's try: randomize U and V with sparse entries, then solve for W.
    
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        
        # Initialize U and V sparsely
        U = np.zeros((U_dim, target_rank))
        V = np.zeros((V_dim, target_rank))
        
        # Each row of U should have a few nonzeros
        for i in range(U_dim):
            nnz = max(1, target_rank // 10)
            cols = np.random.choice(target_rank, size=min(nnz, target_rank), replace=False)
            U[i, cols] = np.random.choice([-1, 1], size=len(cols))
        
        for j in range(V_dim):
            nnz = max(1, target_rank // 10)
            cols = np.random.choice(target_rank, size=min(nnz, target_rank), replace=False)
            V[j, cols] = np.random.choice([-1, 1], size=len(cols))
        
        # Build linear system for W
        # Unknowns: W[k*n+i, r] for all k,i,r -> W_dim * target_rank unknowns
        # Equations: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Number of equations: n*m*p
        # Number of unknowns: p*n*target_rank
        
        num_unknowns = W_dim * target_rank
        num_eqs = n * m * p
        
        # Build A matrix and b vector
        # This could be large, so let's be careful
        if num_unknowns > 50000 or num_eqs > 50000:
            continue
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            # W[k*n+i, r] is at index (k*n+i)*target_rank + r
                            unknown_idx = (k*n+i) * target_rank + r
                            A[eq_idx, unknown_idx] += coeff
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-6:
            continue
        
        W = W_flat.reshape(W_dim, target_rank)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to reduce rank
    # Theoretical lower bound: max(n*m, m*p, p*n) / something... 
    # For matrix mult, lower bound is max(n, m, p) * something
    # Actually the trivial lower bound is max(n*m, m*p, p*n) but that's not tight.
    # For 4x5x6, let's try ranks from 100 down to see if we can beat 120.
    
    # Try a few target ranks below schoolbook
    for target_rank in range(115, 119):  # try 115-119
        if time.time() - start > DEADLINE_MARGIN:
            break
        result = _try_rank_reduction(n, m, p, target_rank, time_limit=20)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
