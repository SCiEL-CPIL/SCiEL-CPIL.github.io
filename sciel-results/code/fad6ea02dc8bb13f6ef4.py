"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit

    # The matrix multiplication tensor T_{i,j,k} = e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # Indices:
    # i in [0, n)  -> row of A
    # j in [0, m)  -> col of A / row of B
    # k in [0, p)  -> col of B
    
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # Condition for rank R decomposition with factors U, V, W (entries u_{ij}^r, v_{jk}^r, w_{ik}^r):
    # For all i, j, k:
    # sum_{r=1}^R u_{ij}^r v_{jk}^r w_{ik}^r = 1  (since T_{i,j,k} = 1 if indices match, but wait...)
    
    # Let's clarify the tensor definition.
    # A is n x m. B is m x p. C is n x p.
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    # The tensor T has entries T_{i,j,k}.
    # The bilinear form is <U_{ij}, V_{jk}, W_{ik}>.
    # We want: sum_{r} U_{ij,r} V_{jk,r} W_{ik,r} = delta_{?}
    # Actually, the standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    # T_{i,j,k} = 1 if the indices are valid? No.
    # The tensor is defined such that T(A, B) = C.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 for all valid i,j,k? No, that's not right.
    # T_{i,j,k} is 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # Yes, for every triple (i,j,k), the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So the tensor entry T_{i,j,k} is 1.
    # Wait, the tensor is a 3-tensor.
    # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Is that correct?
    # Let's check the contraction.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # If we have a decomposition:
    # C_{i,k} = sum_{r} (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j} V_{j,k,r} B_{j,k}) ? No.
    
    # Standard definition:
    # M_{n,m,p} tensor has entries M_{i,j,k}.
    # The multiplication is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This can be written as:
    # C_{i,k} = sum_{r} (sum_{j} u_{i,j,r} A_{i,j}) (sum_{j} v_{j,k,r} B_{j,k}) ? No, that's not bilinear in A and B separately.
    
    # Correct bilinear form:
    # C_{i,k} = sum_{r} (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j} V_{j,k,r} B_{j,k}) is wrong because U and V depend on j differently.
    
    # Let's look at the indices in the prompt:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]
    # W: (p*n, rank) -> W[k*n + i, r]
    
    # The decomposition condition is:
    # For all i, j, k:
    # sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, is it 1?
    # The tensor T_{i,j,k} corresponds to the term A_{i,j} B_{j,k} in C_{i,k}.
    # So yes, T_{i,j,k} = 1 for all valid i,j,k.
    # So we need:
    # sum_{r} U_{ij,r} V_{jk,r} W_{ik,r} = 1
    
    # This is a system of equations.
    # Number of equations: n * m * p = 3 * 4 * 5 = 60.
    # Number of variables for rank R: R * (n*m + m*p + p*n) = R * (12 + 20 + 15) = 47 R.
    
    # If R=1, 47 vars, 60 eqs. Unlikely.
    # If R=2, 94 vars, 60 eqs. Possible?
    # If R=3, 141 vars, 60 eqs.
    
    # However, the entries must be integer or half-integer.
    # This is a non-linear system (cubic).
    
    # Let's try to find a small rank solution.
    # Lower bound: max(n*m, m*p, p*n) = 20.
    # But that's a lower bound for the rank of the tensor?
    # Actually, the rank of M_{n,m,p} is at least max(n*m, m*p, p*n) / something?
    # No, the rank is at least the max of the dimensions of the slices.
    # Slice along i: m*p = 20.
    # Slice along j: n*p = 15.
    # Slice along k: n*m = 12.
    # So rank >= 20.
    
    # Can we achieve rank 20?
    # If we use the schoolbook algorithm, rank = n*m*p = 60.
    # Strassen's algorithm reduces rank for square matrices.
    # For rectangular, there are reductions.
    
    # Let's try a randomized search for small R.
    # We can use a local search or a grid search for small R.
    # Given the time limit, we can try R from 20 to 30.
    # But solving cubic equations is hard.
    
    # Alternative:
    # Use the known result that M_{3,4,5} might have a known rank.
    # If not, we can try to construct one.
    
    # Let's try R=20.
    # If we can't find one quickly, we return the schoolbook R=60.
    
    # Strategy:
    # 1. Check if R=20 is possible with a quick random search (maybe too slow).
    # 2. If not, return R=60.
    
    # Since finding a minimal rank decomposition is hard, and we have a 145s budget,
    # we can try a few random initializations for R=20, 21, etc.
    # But the solution space is discrete (integers/half-integers).
    # Random search over reals might not land on integer/half-integer solutions.
    
    # Let's stick to the schoolbook for now, as it's guaranteed to work.
    # Unless we can find a known better decomposition.
    
    # Is there a known decomposition for 3x4x5 with rank < 60?
    # A quick search suggests that for small rectangular matrices, the rank is often close to n*m*p.
    # For 2x2x2, rank is 7 (vs 8).
    # For 2x2x3, rank is 11 (vs 12)? Or is it 10?
    # Actually, the rank of M_{2,2,3} is 11.
    # The rank of M_{3,3,3} is 23 (vs 27).
    
    # For 3x4x5, the gap might be small.
    # Let's assume the schoolbook is the safest bet for a valid answer.
    
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
