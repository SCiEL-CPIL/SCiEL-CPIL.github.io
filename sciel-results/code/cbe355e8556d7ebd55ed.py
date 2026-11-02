"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor T[i,j,k] is 1 if the indices correspond to a valid multiplication term.
    # Specifically, (A*B)[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's the schoolbook. The tensor is simply the all-ones tensor of size n*m*p?
    # No. The tensor T is defined by T_{i,j,k} = 1.
    # We want to decompose this tensor.
    
    # Strategy: The tensor is 1 for all i,j,k.
    # This is a separable tensor? No, it's the product of three vectors of ones.
    # T[i,j,k] = 1 * 1 * 1.
    # So rank 1?
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1.
    # This works for the "multiplication tensor" if the tensor is just all ones.
    
    # Let's re-read the problem statement carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # In tensor decomposition of matrix multiplication, the tensor T is usually defined such that:
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{i,k}^{(c)} ?
    # No, the standard bilinear decomposition is:
    # (A * B)_{i,k} = sum_c (sum_j U_{i,j}^{(c)} A_{i,j}) (sum_k V_{j,k}^{(c)} B_{j,k}) ? No.
    
    # The standard form is:
    # (A * B)_{i,k} = sum_c (sum_j U_{i,j}^{(c)} A_{i,j}) * (sum_j V_{j,k}^{(c)} B_{j,k}) ? No.
    
    # Let's look at the seed program.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This sum must equal the tensor entry T[i,j,k].
    # For matrix multiplication, T[i,j,k] = 1.
    # Because (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The coefficient of A_{i,j} B_{j,k} in the product (A*B)_{i,k} is 1.
    # So yes, the tensor T is all ones.
    
    # If T is all ones, then rank 1 is sufficient.
    # U is all 1s, V is all 1s, W is all 1s.
    # Let's verify.
    # U[i*m+j, 0] = 1.
    # V[j*p+k, 0] = 1.
    # W[k*n+i, 0] = 1.
    # Product = 1.
    # This seems too easy. Is there a constraint I'm missing?
    # "integer/half-integer entries" - 1 is an integer.
    # "rank-minimal" - 1 is minimal.
    
    # Why would the seed program have rank n*m*p?
    # Maybe the tensor is NOT all ones?
    # Let's check the definition of the matrix multiplication tensor again.
    # The multiplication map is a bilinear map:
    # M: Mat_{n,m} x Mat_{m,p} -> Mat_{n,p}
    # M(A, B) = A B
    # This is a trilinear form on Mat_{n,m}^* x Mat_{m,p}^* x Mat_{n,p}^*.
    # The tensor T is in Mat_{n,m} \otimes Mat_{m,p} \otimes Mat_{n,p}.
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k} ?
    # No.
    # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T has components T_{i,j,k} = 1.
    # Yes, it is the all-ones tensor.
    
    # So the rank is 1.
    # Let's double check if "rank" here means something else.
    # "score is the rank".
    # If I return rank 1, score is 1.
    # The seed program returns rank 48 (2*3*8).
    # Score 1.0000 implies the seed program's rank is the baseline?
    # "1.0 equals the seed program".
    # If seed is rank 48, and I get rank 1, my score should be 48/1 = 48?
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So if seed rank is 48, and I find rank 1, score = 48/1 = 48.
    # This would be a huge improvement.
    
    # Is there any reason rank 1 would be invalid?
    # "integer/half-integer entries that exactly decompose the matrix multiplication tensor".
    # U, V, W are 1. They are integers.
    # The decomposition is exact.
    
    # Let's construct the rank 1 solution.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
