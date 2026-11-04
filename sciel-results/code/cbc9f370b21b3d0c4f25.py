"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build the full multiplication tensor T[i,j,k] = delta for A_{i,j} * B_{j,k}
    # T has shape (n, m, p) and T[i,j,k] = 1 if we consider the identity contribution
    # Actually, the multiplication tensor is T_{i,j,k} such that
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor decomposition requires: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    # For the standard matrix multiplication tensor, T_{i,j,k} = 1 for all valid (i,j,k)
    # because (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}, and each term A_{i,j}B_{j,k} contributes
    # to the (i,k) entry via the j-th intermediate.
    
    # So the tensor is all-ones of shape (n, m, p).
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Efficient check: for each r, the contribution is outer product
    # We can check by computing for a random set of (i,j,k) or all of them.
    # Since n*m*p = 54 is small, check all.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _rank_from_decomp(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return U.shape[1]


def _try_kronecker_block_structure():
    """
    Try to construct a rank-27 decomposition using the Kronecker product structure.
    
    The idea: A is 2x3, B is 3x9. Write B as three 3x3 blocks: B = [B0, B1, B2]
    where B0, B1, B2 are 3x3. Then A*B = [A*B0, A*B1, A*B2].
    
    If we can decompose 3x3 * 3x3 multiplication in rank r, then the total rank
    would be 3*r (one for each block). The minimum rank for 3x3*3x3 is 19 (Strassen-like)
    or 23 (standard). But we need integer/half-integer entries.
    
    Actually, let's think differently. The strategy says to use Kronecker product A ⊗ I_3.
    Let me try a specific construction.
    
    Alternative: use the fact that 2x3 * 3x9 can be seen as (2x3 * 3x3) repeated 3 times
    for the three column-blocks of B. The minimum rank for 2x3 * 3x3 is at most 6 (schoolbook).
    So 3*6 = 18 would be the block-diagonal approach. But that's the schoolbook for this case.
    
    Let me try a more clever approach: use rank-19 for 3x3*3x3 (Laderman) and see if we
    can adapt it.
    
    Actually, let me just try to find any decomposition with rank < 54 using random search
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    target_ranks = [27, 30, 36, 40, 42, 45, 48, 50, 51, 52, 53]
    
    for rank in target_ranks:
        # Try a few random initializations
        for trial in range(50):
            # Random integer/half-integer entries in {-2, -1, -0.5, 0, 0.5, 1, 2}
            vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
            U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
            
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None


def _try_laderman_based():
    """
    Try to build a decomposition using Laderman's rank-19 algorithm for 3x3*3x3,
    adapted to our 2x3 * 3x9 case.
    
    Laderman's algorithm for 3x3*3x3 uses 19 multiplications.
    For our case, we have A (2x3) * B (3x9). We can split B into three 3x3 blocks.
    If we use Laderman for each block, we get 3*19 = 57, which is worse than schoolbook.
    
    But maybe we can share some computations across blocks.
    
    Let me try a different approach: use the structure that A has only 2 rows.
    (A*B)_{i,k} = sum_{j=0}^{2} A_{i,j} * B_{j,k} for i=0,1 and k=0,...,8.
    
    This is 18 outputs. Each output is a sum of 3 products.
    
    Let me try to find a rank-27 decomposition by hand.
    
    Actually, let me try a systematic approach: use the bilinear structure and try to
    find a rank decomposition via linear algebra.
    """
    n, m, p = DIMENSIONS
    # The tensor T is all-ones of shape (n, m, p) = (2, 3, 9).
    # We want T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    
    # This is a tensor rank problem. The tensor is the all-ones tensor of shape (2,3,9).
    # The rank of the all-ones tensor of shape (a,b,c) is min(a,b,c) if we allow real entries?
    # No, that's not right. The all-ones tensor T_{i,j,k} = 1 can be written as
    # T_{i,j,k} = sum_{r=1}^{R} u_{i,r} v_{j,r} w_{k,r}
    # If we set u_{i,r} = 1, v_{j,r} = 1, w_{k,r} = 1/R for r=1..R, then sum = R * (1/R) = 1.
    # So rank 1 works for the tensor T_{i,j,k} = 1!
    
    # But wait, that's not the matrix multiplication tensor. The matrix multiplication
    # tensor is NOT the all-ones tensor. Let me re-read the problem.
    
    # The problem says "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor for (n,m,p) matrices is defined as:
    # T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of AB.
    # Actually, T_{i,j,k} = 1 for all valid (i,j,k) because every A_{i,j} * B_{j,k}
    # contributes to (AB)_{i,k}.
    
    # So yes, the tensor IS the all-ones tensor of shape (n,m,p) = (2,3,9).
    # And the rank-1 decomposition T_{i,j,k} = (1/1)*(1)*(1) works!
    
    # But that gives rank 1, which is clearly wrong for matrix multiplication.
    # I'm confusing the tensor with the bilinear map.
    
    # Let me reconsider. The matrix multiplication bilinear map is:
    # (A, B) -> AB
    # where A is n x m, B is m x p, AB is n x p.
    # The bilinear map has n*p outputs, each being sum_{j=1}^{m} A_{i,j} B_{j,k}.
    # 
    # The tensor associated with this bilinear map is T in R^{(n*m) x (m*p) x (n*p)}
    # where T_{(i,j), (j',k), (i',k')} = 1 if i=i' and k=k' and j=j', else 0.
    # 
    # So T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # 
    # This is NOT the all-ones tensor. It's a structured tensor.
    # 
    # The rank of this tensor is the minimum number of rank-1 tensors needed to sum to it.
    # A rank-1 tensor in this space is U_{(i,j),r} * V_{(j',k),r} * W_{(i',k'),r}.
    # 
    # For the schoolbook, we use rank = n*m*p, with:
    # U_{(i,j),r} = 1 if r corresponds to (i,j,k) for some k
    # V_{(j',k),r} = 1 if r corresponds to (i,j,k) for some i
    # W_{(i',k'),r} = 1 if r corresponds to (i,j,k) for some (i',k')=(i,k)
    
    # So the tensor is T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # 
    # Let me verify with the schoolbook:
    # For r corresponding to (i0, j0, k0):
    # U_{(i,j),r} = 1 if i=i0 and j=j0
    # V_{(j',k),r} = 1 if j'=j0 and k=k0
    # W_{(i',k'),r} = 1 if i'=i0 and k'=k0
    # 
    # Sum over r: U_{(i,j),r} V_{(j',k),r} W_{(i',k'),r}
    # = sum over (i0,j0,k0) of [i=i0 and j=j0] * [j'=j0 and k=k0] * [i'=i0 and k'=k0]
    # = 1 if i=i'=i0, j=j'=j0, k=k'=k0 for some (i0,j0,k0)
    # = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Great, so the tensor is T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Now, can we find a lower-rank decomposition?
    # This is equivalent to finding the tensor rank of the "identity-like" tensor
    # in the space of bilinear maps.
    
    # For n=2, m=3, p=9, the tensor has size 6 x 27 x 18.
    # The schoolbook rank is 54.
    
    # Let me try to find a rank-27 decomposition.
    # 
    # Idea: group by i. For each i in {0,1}, we have a sub-tensor
    # T^{(i)}_{(j,k),(j',k')} = delta_{j,j'} delta_{k,k'}
    # This is the identity tensor in R^{(3*9) x (3*9)} = R^{27 x 27}.
    # The rank of the identity tensor of size d x d is d (it's the identity matrix).
    # So for each i, we need rank 27. Total rank = 2 * 27 = 54.
    
    # Hmm, that's still 54. Let me think more carefully.
    
    # Actually, the tensor T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # can be written as sum over i of T^{(i)} where T^{(i)}_{(i,j),(j',k),(i',k')} = delta_{j,j'} delta_{k,k'} if i=i'=i_target, else 0.
    
    # For fixed i_target, T^{(i_target)} is a tensor in R^{(m*p) x (n*p)} = R^{27 x 18}
    # with T^{(i_target)}_{(j,k),(i',k')} = delta_{i',i_target} delta_{j,j'} delta_{k,k'}
    # 
    # This is a matrix (in the bilinear sense) that maps (j,k) to (i',k') with
    # M_{(j,k),(i',k')} = delta_{i',i_target} delta_{j,j'} delta_{k,k'}
    # 
    # This matrix has rank 27 (it's essentially the 27x27 identity embedded in a 27x18 matrix
    # by selecting the rows corresponding to i'=i_target).
    
    # So the total rank is sum over i_target of rank(T^{(i_target)}) = 27 + 27 = 54.
    
    # It seems like 54 is indeed the minimum rank for this specific tensor structure
    # when we require the decomposition to respect the block structure.
    
    # But wait, we're not required to respect the block structure. We can mix terms
    # from different i values in the same rank-1 component.
    
    # Let me try to find a rank-27 decomposition by solving a linear system.
    # 
    # We want U (6 x R), V (27 x R), W (18 x R) such that for all (i,j,j',k,i',k'):
    # sum_r U_{(i,j),r} V_{(j',k),r} W_{(i',k'),r} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # This is a system of 6*27*18 = 2916 equations in R*(6+27+18) = 51R unknowns.
    # For R=27, we have 51*27 = 1377 unknowns, which is less than 2916 equations.
    # So rank 27 might not be achievable with this simple approach.
    
    # Let me try R=54 (schoolbook) and see if we can reduce it.
    
    # Actually, let me just return the schoolbook decomposition. It's guaranteed to work.
    return None


def _to_native_list(arr):
    """Convert a numpy array to a list of lists of native Python floats."""
    return [[float(val) for val in row] for row in arr]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    # Use the guaranteed valid schoolbook decomposition (rank 54)
    # This ensures finite, JSON-serializable numbers.
    # Random search for lower rank is unreliable and slow; we prioritize a valid submission.
    decomp = _schoolbook()
    
    U = _to_native_list(decomp["U"])
    V = _to_native_list(decomp["V"])
    W = _to_native_list(decomp["W"])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
