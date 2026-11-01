import numpy as np
import itertools
import time
import sys

DIMENSIONS = [2, 2, 4]

def _verify_rank(n, m, p, rank):
    """
    Attempts to find a decomposition of rank 'rank' for matrix multiplication n x m * m x p.
    Returns (U, V, W) if found, else None.
    Uses a heuristic search: 
    1. Fix U to be the identity-like basis (or a simple structured matrix) to reduce variables? 
       Actually, we can't just fix U because the decomposition is not unique.
       
    However, for small ranks, we can try to solve the system of polynomial equations.
    The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    
    Since n=2, m=2, p=4, the tensor has 2*2*4 = 16 components.
    The variables are U (4xR), V (8xR), W (8xR). Total 20R variables.
    
    A simpler approach for small dimensions:
    We can iterate over possible structures. 
    For rank R, we can assume a "standard form" if we are lucky, but generally we need to search.
    
    Given the time constraint (150s) and the small size, we can try a randomized local search or a structured enumeration.
    However, a more robust approach for "minimal rank" in small cases is often to check known bounds or use a solver.
    
    Let's try a randomized search with verification.
    We generate random U, V, W with entries from a small set (e.g., {0, 1, -1} or fractions).
    Then we check if they satisfy the tensor equation.
    If we find one, we return it.
    
    To make it deterministic and fast, we can try:
    1. Rank 8 (theoretical lower bound for 2x2*2x2 is 7, for 2x2*2x4 it's likely 8 or 9).
    2. If not found quickly, try Rank 9, 10, etc.
    
    But random search might be slow. Let's look at the structure.
    Matrix multiplication M_{i,j} = sum_k A_{i,k} B_{k,j}.
    Here A is 2x2, B is 2x4.
    
    Actually, there is a known result for small matrix multiplication.
    The trivial rank is 16.
    Strassen's algorithm for 2x2 gives rank 7 for 2x2*2x2.
    For 2x2 * 2x4, we can view it as two 2x2 multiplications side-by-side.
    So rank <= 7 + 7 = 14.
    Can we do better?
    
    Let's try to find a rank 12 or 13 decomposition using a structured search.
    Or, we can use the fact that the problem is small enough to run a SAT/SMT solver if available, but we only have numpy/scipy.
    
    Let's implement a randomized search with a tight deadline.
    We will try ranks from 8 upwards.
    For each rank, we will run a limited number of random trials.
    """
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the product term A[i,j] * B[j,k] contributes? 
    # No, the tensor for matrix multiplication is defined such that 
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The trilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]? 
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the contraction is: sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{ik} ? 
    # No, it should equal the coefficient of the product.
    # Usually, the tensor M has entries M[i,j,k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    # In the trace convention for the bilinear form:
    # sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_{i,j} (sum_k A[i,j] B[j,k]) C[k,i] 
    # Wait, C[k,i] is not standard. Standard is C[i,k].
    # If the convention is W[k*n+i], then the index is k*n+i.
    # The sum is sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This sums over j and r. The free indices are i and k.
    # The result should be 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # In standard matrix multiplication C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T[i,k,j] (if we map W to i,k) should be 1 for all j?
    # No, the decomposition is:
    # T[i, k, j] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # And T[i, k, j] should be 1 if the product A[i,j] B[j,k] is part of the sum for C[i,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], every j contributes.
    # So T[i,k,j] = 1 for all i,k,j?
    # Let's check the dimensions.
    # U: (n*m, rank) -> indices (i, j)
    # V: (m*p, rank) -> indices (j, k)
    # W: (p*n, rank) -> indices (k, i)
    # Sum over j, r: U[i,j,r] V[j,k,r] W[k,i,r]
    # This equals 1 for all i, k, j?
    # If so, the tensor is the all-ones tensor of size n x p x m.
    # Let's verify with the trivial decomposition.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over r: only one r is non-zero for a specific (i,j,k) triple?
    # In the trivial decomposition, each (i,j,k) has its own column r.
    # So for a fixed (i,j,k), the term is 1*1*1 = 1.
    # For any other combination, it's 0.
    # So the resulting tensor T[i,k,j] is 1 if (i,j,k) is a valid index?
    # Yes, the tensor of matrix multiplication is indeed the "all ones" tensor in the sense that
    # every combination of (i,j,k) corresponds to a multiplication term.
    # Wait, is it?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is <A, B, C> = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # If we use the trace convention W[k*n+i], then C is indexed by k,n+i?
    # If C is a matrix, C[i,k] is the element.
    # The prompt says W[k*n+i]. This suggests the index is k*n+i.
    # If n=2, p=4.
    # If C is 2x4, C[i,k] has i in 0..1, k in 0..3.
    # Index k*n+i = k*2+i.
    # This is a standard row-major or column-major flattening?
    # If C is stored as C[i][k], index is i*p+k.
    # If C is stored as C[k][i] (transpose), index is k*n+i.
    # So the "output factor" W corresponds to the transpose of C?
    # Or C is defined such that C[k,i] is the element?
    # Let's assume the tensor T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m).
    # Because every term A[i,j] B[j,k] contributes to C[i,k] (or C[k,i]).
    
    # So we need to decompose the all-ones tensor of size n x p x m into rank R.
    # T[i,k,j] = 1.
    
    # This is a specific tensor.
    # We want to find min R such that T = sum_r u_r v_r w_r.
    # Where u_r is a vector of length n*m (flattened i,j), v_r length m*p (flattened j,k), w_r length p*n (flattened k,i).
    # Wait, the indices in the sum are:
    # Sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is not a simple outer product of vectors. It's a contraction.
    # Let's rewrite.
    # Let U_r be a matrix of size n x m? No, U is (n*m, R).
    # So U[:, r] is a vector of length n*m. Let's call it u_r.
    # v_r is a vector of length m*p.
    # w_r is a vector of length p*n.
    # The term is sum_j u_r[i,j] v_r[j,k] w_r[k,i].
    # This is the trace of (u_r as matrix) * (v_r as matrix) * (w_r as matrix)?
    # Let U_r be n x m, V_r be m x p, W_r be p x n.
    # Then sum_j U_r[i,j] V_r[j,k] W_r[k,i] = Tr(U_r V_r W_r).
    # So we need to decompose the all-ones tensor T such that
    # T[i,k,j] = 1 = sum_r Tr(U_r V_r W_r) ?
    # No, the equation is:
    # 1 = sum_r (sum_j U_r[i,j] V_r[j,k] W_r[k,i])
    # This must hold for ALL i, k, j.
    
    # This is a system of equations.
    # For small R, we can try to find integer/half-integer solutions.
    
    # Strategy:
    # 1. Try R=8, 9, 10...
    # 2. For each R, use a randomized search with local refinement.
    # 3. Since the tensor is all-ones, there might be symmetric solutions.
    
    # Let's try a specific structured ansatz.
    # If we use the Strassen-like decomposition for 2x2, we get rank 7.
    # For 2x2 * 2x4, we can treat it as two 2x2 multiplications.
    # Rank 14 is guaranteed.
    # Can we do 13? 12?
    
    # Given the time limit, I will implement a randomized search for R=12 to 14.
    # If it finds a solution, it returns it.
    # If not, it returns the trivial R=16.
    
    # To speed up verification, we can check the tensor equation.
    
    def check_decomp(U, V, W):
        n, m, p = 2, 2, 4
        # T[i,k,j] should be 1
        # Sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Randomized search
    # We will use a small set of values for entries: {0, 1, -1, 0.5, -0.5}
    # This is a heuristic.
    
    # Actually, a better approach for "all-ones" tensor:
    # The tensor is rank 1 if it were all ones in a single dimension, but it's 3D.
    # The rank of the all-ones tensor of size A x B x C is not trivially 1.
    # Wait, if T[i,k,j] = 1, then T = 1*1*1?
    # No, the indices are coupled.
    # T[i,k,j] = 1.
    # If we take u_r[i,j] = 1, v_r[j,k] = 1, w_r[k,i] = 1.
    # Then sum_j 1*1*1 = m = 2.
    # So if we take R=1, and scale by 1/2, we get 1.
    # U[i*m+j, 0] = 1/2
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Then sum_j (1/2)*1*1 = 2*(1/2) = 1.
    # So Rank 1 is sufficient?
    
    # Let's re-read the tensor definition.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 4 matrix."
    # The tensor of matrix multiplication is NOT the all-ones tensor.
    # The tensor M is defined by M[i,j,k] = 1 if the product A[i,j]B[j,k] appears in C[i,k].
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j]B[j,k].
    # So for a fixed i,k, ALL j contribute.
    # So M[i,k,j] = 1 for all i,k,j.
    # So yes, it IS the all-ones tensor.
    
    # If it is the all-ones tensor, then Rank 1 is sufficient!
    # U[i*m+j, 0] = 1/m (1/2)
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Check: sum_j (1/2) * 1 * 1 = 2 * 1/2 = 1.
    # This works for all i,k,j.
    
    # Why is the trivial rank 16 then?
    # Because the trivial decomposition uses the standard basis.
    # But the rank is the MINIMUM number of terms.
    # If Rank 1 works, the answer is 1.
    
    # Let's double check the "trace convention" and indices.
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    # Equation: sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    
    # If R=1:
    # U[i*m+j, 0] = 0.5
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum over j (0 to 1): 0.5 * 1 * 1 = 0.5.
    # Sum is 0.5 + 0.5 = 1.
    # Yes!
    
    # So the minimal rank is 1.
    # The entries are half-integers (0.5, 1). This is allowed ("integer/half-integer entries").
    
    return {
        "U": [[0.5] * 1 for _ in range(n*m)],
        "V": [[1] * 1 for _ in range(m*p)],
        "W": [[1] * 1 for _ in range(p*n)]
    }

def construct():
    n, m, p = DIMENSIONS
    # Based on the analysis, rank 1 is sufficient for the all-ones tensor.
    # Let's verify if the tensor is indeed all-ones.
    # Matrix multiplication C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that the bilinear form is sum T[...] A B C.
    # The prompt implies the decomposition of the "multiplication tensor".
    # The multiplication tensor has a 1 for every (i,j,k) triple that corresponds to a valid multiplication step.
    # Since every step is valid, it is all-ones.
    
    # However, sometimes "tensor decomposition" refers to the structural tensor where
    # T[i,j,k] is the coefficient.
    # If the task implies finding the rank of the matrix multiplication tensor,
    # and the tensor is all-ones, the rank is 1.
    
    # Is there any catch?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This just defines the index mapping.
    
    # Let's provide the Rank 1 solution.
    U = [[0.5] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
