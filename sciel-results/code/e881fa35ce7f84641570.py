"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        # Compute via decomposition: C[i,k] = sum_c U[i*m + ?, c] ... 
        # Actually the standard form: C[i,k] = sum_j sum_c (U[i,j,c] * V[j,k,c] * W[i,k,c])
        # But our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
        # So C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _try_find_lower_rank(n, m, p, deadline, start_rank):
    """Try to find a lower rank decomposition than schoolbook."""
    best = None
    best_rank = n * m * p
    
    # Strategy 1: Try to find a decomposition by exploiting structure.
    # For matrix multiplication, the tensor is T[i,j,k] = delta (i.e., 1 if indices match appropriately)
    # Actually T is defined by: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor M[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    # We need rank-r decomposition: M[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    
    # Let's try a structured search: start with schoolbook and try to merge columns.
    # Two columns c1, c2 can be merged if there exist vectors such that
    # U_c1 * V_c1 * W_c1 + U_c2 * V_c2 * W_c2 = U_new * V_new * W_new
    # This is generally hard, but for special cases (like when two terms differ in only one index)
    # we can sometimes combine them.
    
    # A simpler approach: use the fact that for the 4x4x6 case, we can try
    # to find a decomposition with rank < 96 by solving a system of equations.
    
    # Let's try random search with verification for a few ranks.
    # This is unlikely to find a good answer quickly, but let's try.
    
    for target_rank in range(start_rank, best_rank):
        if time.time() > deadline:
            break
        # Skip if target_rank < theoretical lower bound
        if target_rank < max(n*m, m*p, p*n):
            continue
        
        # Try a few random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(attempt + target_rank)
            # Initialize with small random values
            U_rand = rng.randn(n * m, target_rank) * 0.1
            V_rand = rng.randn(m * p, target_rank) * 0.1
            W_rand = rng.randn(p * n, target_rank) * 0.1
            
            # This won't converge to an exact solution easily, so let's just
            # try to verify if by chance it works (very unlikely).
            # Instead, let's use a different strategy.
            pass
    
    # Strategy 2: Use known algebraic identities for small matrices.
    # For 4x4x6, let's see if we can use the fact that the 6 dimension is larger.
    # One approach: split the 6 columns into two groups of 3 and use 
    # a decomposition that shares some factors.
    
    # Actually, let's try a more systematic approach: 
    # The multiplication tensor for n x m * m x p has rank at most n*m*p.
    # We can try to find a lower rank by using the Strassen-like approach
    # or by finding linear dependencies among the rank-1 terms.
    
    # For now, let's just return the schoolbook if we can't find better quickly.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But matrix multiplication rank is known to be higher.
    # For 4x4x4, the best known is 49 (Landsberg). For 4x4x6, it's unknown
    # but likely around 60-80.
    
    # Let's try a structured search: attempt to merge schoolbook terms.
    # Key idea: if two terms (i1,j1,k1) and (i2,j2,k2) have i1=i2, j1=j2, k1!=k2,
    # then we can potentially combine them using the identity:
    # a*b*c1 + a*b*c2 = a*b*(c1+c2) if the W factors align.
    # More generally, we can look for linear combinations that reduce rank.
    
    # For a 4x4x6 matrix multiplication, let's try to exploit the rectangular shape.
    # One approach: use the fact that we can group terms by (i,j) and share
    # the U factor, then find a low-rank decomposition of the remaining 6x6 structure.
    
    # Actually, the cleanest approach: try to find a decomposition by solving
    # for the factors directly. Let's use the following strategy:
    # 
    # We want M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where M[i,j,k] = 1 for all i,j,k.
    # 
    # This is equivalent to: for each (i,k), sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # 
    # Let's try to find a solution with rank r by using a structured ansatz.
    
    # Ansatz: Try to use a decomposition where the rank is n*m + m*p + p*n - (n-1)*(m-1)*(p-1)
    # or similar. But this is getting complex.
    
    # Let's just do a quick search and return the best we find.
    # For safety, return schoolbook if nothing better is found.
    
    # Try a few specific rank values with structured search
    for target_rank in [72, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24]:
        if time.time() > deadline:
            break
        if target_rank < max(n*m, m*p, p*n):
            continue
        
        # Try to find a valid decomposition with this rank
        # Use a simple iterative approach: start with random init and try to
        # minimize the error, checking for exactness periodically.
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed + target_rank * 1000)
            
            # Initialize factors
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # Try a few gradient steps to reduce error
            # Error = sum_{i,j,k} (M[i,j,k] - sum_c U[i,j,c]V[j,k,c]W[i,k,c])^2
            # This is a 3-tensor completion problem.
            
            # For simplicity, let's just check if a random init happens to work
            # (extremely unlikely) or use a structured approach.
            
            # Structured approach: use the fact that M is all-ones.
            # If we set U[i*m+j, c] = a_c, V[j*p+k, c] = b_c, W[k*n+i, c] = c_c
            # then sum_c a_c * b_c * c_c must equal 1 for all (i,j,k).
            # This requires that the sum is independent of (i,j,k), which means
            # the factors must be constant across i,j,k for each c.
            # But then U[i*m+j, c] = a_c for all i,j, which means U has rank 1
            # in the (i,j) index. Similarly for V and W.
            # So we need: sum_c a_c * b_c * c_c = 1, with a_c, b_c, c_c constants.
            # This gives us a rank-1 decomposition of the all-ones tensor!
            # But wait, that's not right - the multiplication tensor isn't all-ones
            # in the sense I'm thinking. Let me reconsider.
            
            # Actually, the multiplication tensor T is defined by T[i,j,k] = 1 for all
            # i,j,k, and the multiplication is: C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            # So T[i,j,k] = 1 for all i,j,k. Yes, it's the all-ones tensor!
            # 
            # So we need: 1 = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] for all i,j,k.
            # 
            # If we can find U, V, W such that this holds, we're done.
            # The trivial solution is U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1
            # for c=0, giving rank 1. But that's not right because the factors
            # need to be indexed correctly.
            # 
            # Wait, I think I'm confusing myself. Let me re-examine.
            # 
            # The standard tensor decomposition for matrix multiplication is:
            # C[i,k] = sum_c sum_j U[i,j,c] V[j,k,c] W[i,k,c] A[i,j] B[j,k]
            # For this to equal sum_j A[i,j] B[j,k], we need:
            # sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1 for all i,j,k.
            # 
            # So yes, T[i,j,k] = 1 for all i,j,k. The tensor is all-ones.
            # 
            # Now, can we decompose the all-ones tensor of size (n*m) x (m*p) x (p*n)
            # into a sum of rank-1 tensors? The rank of the all-ones tensor is 1!
            # Because 1 = 1 * 1 * 1.
            # 
            # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1.
            # But that can't be right because matrix multiplication has rank > 1.
            # 
            # I think the issue is that the tensor isn't all-ones in the way I'm thinking.
            # Let me reconsider the definition.
            
            # Actually, I realize my mistake. The multiplication tensor is NOT all-ones.
            # The correct tensor for matrix multiplication is:
            # T[i, j, k] = delta_{appropriate}
            # 
            # No wait, for matrix multiplication C = A * B where A is n x m and B is m x p:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # 
            # The tensor T is defined such that C[i,k] = sum_{j,c} T[i,j,k,c] A[i,j] B[j,k]
            # where T[i,j,k,c] = 1 if this term contributes.
            # 
            # Actually, the standard form is:
            # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j' V[j',k,c] B[j',k]) W[i,k,c]
            # 
            # No, that's not right either. Let me look at this more carefully.
            # 
            # The rank of matrix multiplication is the minimum number of multiplications
            # needed to compute C = A * B. Each multiplication corresponds to a rank-1
            # term in the tensor decomposition.
            # 
            # The multiplication tensor M is a 3-tensor with indices (i, j, k) where
            # i in [0,n), j in [0,m), k in [0,p), and M[i,j,k] = 1 for all i,j,k.
            # 
            # A rank-r decomposition is: M[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] V[j,k,c] W[i,k,c]
            # 
            # For this to work, we need: sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1 for all i,j,k.
            # 
            # The schoolbook solution uses r = n*m*p, with each term corresponding to
            # a specific (i,j,k) triple.
            # 
            # Now, can we do better? The rank of the all-ones tensor of size a x b x c
            # is 1, because 1 = 1*1*1. But that would mean matrix multiplication has
            # rank 1, which is clearly wrong.
            # 
            # I think the resolution is that the tensor isn't all-ones in the (i,j,k)
            # space in the way I'm thinking. Let me reconsider.
            # 
            # Actually, I think the issue is that U, V, W are not arbitrary functions
            # of (i,j), (j,k), (i,k). They must be such that the decomposition is
            # valid for ALL A and B, not just specific ones.
            # 
            # Let me think about this differently. The multiplication tensor is:
            # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (in some appropriate space)
            # 
            # No, that's not right. The multiplication tensor is actually:
            # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  where the sum is over all valid indices.
            # 
            # I'm going in circles. Let me just accept that the schoolbook rank is n*m*p
            # and try to find a lower rank through search.
            
            pass
    
    # Given the complexity of finding a lower rank decomposition, and the time constraints,
    # let's just return the schoolbook decomposition as the best we can guarantee.
    return baseline
# EVOLVE-BLOCK-END
