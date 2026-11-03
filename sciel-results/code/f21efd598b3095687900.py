"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try rank 55 first, fall back to 60
    for rank in [55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        try:
            result = _try_rank(n, m, p, rank)
            if result is not None:
                return result
        except Exception:
            pass
    
    # Fallback: schoolbook rank 60
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

def _try_rank(n, m, p, rank):
    """Try to construct a rank-`rank` decomposition."""
    # Use a Strassen-like approach for the 2x2 sub-block
    # For 2x3x10: split j into {0,1} and {2}
    # j=2: 20 terms (one per (i,k))
    # j in {0,1}: need 2x2x10 tensor decomposition
    
    # For 2x2x10, use rank = 7*10 - 6*9 = 70-54 = 16? No, that's not right.
    # Let me use a simpler approach: 2x2xp has rank at most 4p, and with Strassen savings,
    # maybe 7p - 3(p-1) = 4p+3? For p=10, that's 43.
    # 43 + 20 = 63. Still > 55.
    
    # Alternative: use a global decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Try a structured decomposition
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Use the following construction:
    # For each (i,k), we need the sum over j of A[i,j]*B[j,k] to be captured.
    # Idea: use rank-1 terms for each (i,k) with j-dimension handled by coefficients.
    
    # Simple approach: for each (i,k), use one term with U[i,j,r] = 1 for all j,
    # V[j,k,r] = 1 for all j, W[k,i,r] = 1. But this gives T[i,j,k] = 1 for all j,
    # which is correct! But it uses n*p = 20 terms, and T[i,j,k] = sum over r of 1*1*1 = 1
    # only if there's exactly one r for each (i,k). But then for a fixed (i,j), multiple k's
    # contribute, so U[i,j,r] would need to be 1 for multiple r's, and the sum would be > 1.
    
    # The issue: for fixed (i,j), T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must = 1 for all k.
    # If I use 20 terms (one per (i,k)), then for term r corresponding to (i',k'),
    # U[i,j,r] is nonzero only if i'=i, V[j,k,r] nonzero only if k=k', W[k,i,r] nonzero only if k=k' and i=i.
    # So for T[i,j,k], only the term r=(i,k) contributes: U[i,j,r]*V[j,k,r]*W[k,i,r] = 1*1*1 = 1. 
    # But wait, U[i,j,r] for r=(i,k) - is it 1 for all j? If so, then for T[i,j,k'], where k'≠k,
    # the term r=(i,k) has V[j,k',r] = 0 (since k'≠k), so it doesn't contribute. Good.
    # So 20 terms suffice? Let me verify:
    # T[i,j,k] = sum_{r} U[i,j,r]*V[j,k,r]*W[k,i,r]
    # With r indexed by (i',k'): U[i,j,(i',k')] = 1 if i'=i else 0
    # V[j,k,(i',k')] = 1 if k'=k else 0
    # W[k,i,(i',k')] = 1 if i'=i and k'=k else 0
    # So T[i,j,k] = U[i,j,(i,k)]*V[j,k,(i,k)]*W[k,i,(i,k)] = 1*1*1 = 1. YES!
    # So rank 20 suffices?!
    
    # Wait, that can't be right. Let me double-check with the schoolbook: rank 60.
    # The difference: in schoolbook, each (i,j,k) triple has its own term. In my construction,
    # each (i,k) pair has one term that works for all j. Let me verify the tensor equation.
    
    # The tensor equation for matrix multiplication is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T encodes this as: C[i,k] = sum_{i,j,k} T[i,j,k]*A[i,j]*B[j,k]
    # where T[i,j,k] = 1 if (i,j,k) is a valid index triple.
    
    # In the decomposition: C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
    # Wait, that's not right either. The decomposition is:
    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k])
    
    # No, the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] summed over j and r.
    # More precisely: C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    
    # So the condition is: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    # With my 20-term construction:
    # U[i,j,(i',k')] = delta_{i,i'}
    # V[j,k,(i',k')] = delta_{k,k'}
    # W[k,i,(i',k')] = delta_{i,i'}*delta_{k,k'}
    # Then sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = U[i,j,(i,k)]*V[j,k,(i,k)]*W[k,i,(i,k)] = 1*1*1 = 1.
    # This works! So rank 20 is sufficient?!
    
    # But wait, that would mean M_{2,3,10} has rank 20, which seems too low.
    # Let me check: for M_{2,2,2}, rank is 7. With my construction, rank would be 2*2=4.
    # But 4 < 7, which contradicts the known result. So I'm making an error.
    
    # The error: in the tensor decomposition, the indices must be consistent.
    # U is indexed by (i,j), V by (j,k), W by (k,i). The "j" index in U and V must be the SAME j.
    # In my construction, U[i,j,(i',k')] = delta_{i,i'} doesn't depend on j, so for a fixed (i,j),
    # U[i,j,r] is the same for all j. That's fine.
    # V[j,k,(i',k')] = delta_{k,k'} doesn't depend on j. That's fine.
    # W[k,i,(i',k')] = delta_{i,i'}*delta_{k,k'} doesn't depend on j. That's fine.
    
    # So sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_{i',k'} delta_{i,i'}*delta_{k,k'}*delta_{i,i'}*delta_{k,k'}
    # = delta_{i,i}*delta_{k,k} = 1. This is correct!
    
    # So why is M_{2,2,2} rank 7 and not 4? Let me re-examine.
    # For M_{2,2,2}: n=2, m=2, p=2. U is 4×R, V is 4×R, W is 4×R.
    # My construction: R=4 (one per (i,k) pair, i∈{0,1}, k∈{0,1}).
    # U[i,j,(i',k')] = delta_{i,i'}
    # V[j,k,(i',k')] = delta_{k,k'}
    # W[k,i,(i',k')] = delta_{i,i'}*delta_{k,k'}
    # Check: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = U[i,j,(i,k)]*V[j,k,(i,k)]*W[k,i,(i,k)] = 1.
    # This gives rank 4 for M_{2,2,2}, but the known rank is 7. Contradiction!
    
    # I must be misunderstanding the tensor decomposition format. Let me re-read the task.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T ∈ F^{(n*m) × (m*p) × (p*n)} where
    # T[i*m+j, j*p+k, k*n+i] = 1 and T[a,b,c] = 0 otherwise.
    # The decomposition is T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    # So the condition is: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
    # for all valid (i,j,k), and T[a,b,c] = 0 for all other (a,b,c).
    
    # The key constraint I missed: T[a,b,c] = 0 for invalid (a,b,c) triples!
    # For example, T[0, 10, 1] should be 0 (since a=0 means i=0,j=0, b=10 means j=1,k=0, but j≠0).
    # So I need: for all (a,b,c) where the indices don't match up, sum_r U[a,r]*V[b,r]*W[c,r] = 0.
    
    # This is the crucial constraint I was missing! My 20-term construction doesn't satisfy
    # T[a,b,c] = 0 for mismatched indices.
    
    # For example, with my construction:
    # T[0*3+0, 1*10+0, 0*2+1] = T[0, 10, 1]
    # = sum_r U[0,r]*V[10,r]*W[1,r]
    # U[0,r] = delta_{0,i'} where r=(i',k'), so U[0,r] = 1 if i'=0
    # V[10,r] = delta_{0,k'} where r=(i',k'), so V[10,r] = 1 if k'=0
    # W[1,r] = delta_{0,i'}*delta_{0,k'} where r=(i',k'), so W[1,r] = 1 if i'=0 and k'=0
    # So T[0,10,1] = U[0,(0,0)]*V[10,(0,0)]*W[1,(0,0)] = 1*1*1 = 1 ≠ 0. WRONG!
    
    # So my construction is invalid. The zero constraints are essential.
    
    return None  # Can't easily construct a valid low-rank decomposition

# EVOLVE-BLOCK-END
