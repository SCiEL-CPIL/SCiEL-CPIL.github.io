"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import random
import itertools

def construct():
    n, m, p = DIMENSIONS
    # The task is to find a rank-minimal exact tensor decomposition.
    # The schoolbook rank is n*m*p = 48.
    # We will attempt to find a decomposition with rank < 48 using random linear combinations
    # and checking if the resulting tensor matches the matrix multiplication tensor.
    
    # First, let's define the tensor T such that T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k].
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # is defined by the bilinear map (A, B) -> C.
    # The tensor T in the decomposition U, V, W satisfies:
    # C[i, k] = sum_r U[i, r] * V[r, k] * W[i, k] ... wait, the interface is:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # This is a bit unusual. Let's re-read the contract.
    # "U', 'V', 'W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # Let's check the standard tensor decomposition for matrix multiplication.
    # Usually, we have T_{ijk} where i indexes rows of A, j indexes columns of A/rows of B, k indexes columns of B.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as a sum of rank-1 tensors:
    # C_{ik} = sum_r (sum_j A_{ij} u_{ijr}) (sum_j B_{jk} v_{jkr}) w_{ikr} ? No, that's not right.
    
    # Let's look at the shapes:
    # U: (n*m, rank). Index i*m+j corresponds to A_{ij}.
    # V: (m*p, rank). Index j*p+k corresponds to B_{jk}.
    # W: (p*n, rank). Index k*n+i corresponds to C_{ik} (based on "W[k*n+i,r]").
    
    # So the decomposition is:
    # C_{ik} = sum_r W_{k*n+i, r} * ( sum_j U_{i*m+j, r} * A_{ij} ) * ( sum_j V_{j*p+k, r} * B_{jk} ) ?
    # No, that would be trilinear. The standard bilinear decomposition is:
    # C_{ik} = sum_r ( U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} ) ? No.
    
    # Let's assume the standard form:
    # C_{ik} = sum_r ( sum_j A_{ij} u_{ijr} ) ( sum_j B_{jk} v_{jkr} ) w_{ikr} is not correct because the sums are over j.
    
    # Let's reconsider. A rank-1 term in the tensor decomposition of matrix multiplication is:
    # C_{ik} += (sum_j A_{ij} a_{ij}) (sum_j B_{jk} b_{jk}) c_{ik}
    # This is not linear in A and B separately if a and b depend on j in a coupled way.
    # Actually, the standard rank-1 tensor for matrix multiplication is:
    # C_{ik} += A_{ij} B_{jk} C_{ik} for fixed i, j, k. This is the schoolbook.
    
    # The general form is:
    # C_{ik} = sum_r ( sum_j A_{ij} u_{ijr} ) ( sum_j B_{jk} v_{jkr} ) w_{ikr}
    # This requires that the "rank-1" term is a product of three vectors, one for A, one for B, one for C.
    # But the vector for A is indexed by (i,j), for B by (j,k), for C by (i,k).
    # So, for a fixed r:
    # A_part = sum_j A_{ij} u_{ijr}  (this is a scalar for each i? No, it depends on i and j)
    # Wait, the output C_{ik} depends on i and k.
    # So, C_{ik} = sum_r ( sum_j A_{ij} u_{ijr} ) ( sum_j B_{jk} v_{jkr} ) w_{ikr}
    # This formula implies that for a fixed r, we compute a scalar for each i from A, a scalar for each k from B, and then multiply them by w_{ikr} and sum over r.
    # But the scalar from A depends on i, and the scalar from B depends on k.
    # Let's check if this matches the schoolbook.
    # In schoolbook, for each i, j, k: C_{ik} += A_{ij} B_{jk}.
    # This can be seen as r = (i, j, k).
    # u_{ijr} = 1 if r=(i,j,k) else 0.
    # v_{jkr} = 1 if r=(i,j,k) else 0.
    # w_{ikr} = 1 if r=(i,j,k) else 0.
    # Then sum_j A_{ij} u_{ijr} = A_{ij} (if r=(i,j,k)).
    # sum_j B_{jk} v_{jkr} = B_{jk} (if r=(i,j,k)).
    # Product: A_{ij} B_{jk} * 1.
    # Sum over r: sum_{i',j',k'} A_{ij'} B_{j'k'} delta_{ii'}delta_{jj'}delta_{kk'} = A_{ij} B_{jk}.
    # Yes, this works.
    
    # So the decomposition is:
    # C_{ik} = sum_r ( sum_j A_{ij} U_{i*m+j, r} ) ( sum_j B_{jk} V_{j*p+k, r} ) W_{k*n+i, r}
    
    # We want to find U, V, W with minimal rank r such that this identity holds for all A, B.
    # This is equivalent to finding a rank-r decomposition of the matrix multiplication tensor.
    
    # We can try to find such a decomposition by solving a system of linear equations.
    # For each i, k, the coefficient of A_{ij} B_{j'k'} in the expression must be 1 if j=j' and 0 otherwise.
    # The expression is:
    # sum_r ( sum_j A_{ij} U_{i*m+j, r} ) ( sum_{j'} B_{j'k} V_{j'*p+k, r} ) W_{k*n+i, r}
    # = sum_r W_{k*n+i, r} ( sum_j A_{ij} U_{i*m+j, r} ) ( sum_{j'} B_{j'k} V_{j'*p+k, r} )
    # = sum_{j, j'} A_{ij} B_{j'k} ( sum_r U_{i*m+j, r} V_{j'*p+k, r} W_{k*n+i, r} )
    
    # So we require:
    # sum_r U_{i*m+j, r} V_{j'*p+k, r} W_{k*n+i, r} = delta_{j, j'}
    
    # This is a system of equations for the unknowns U, V, W.
    # The number of unknowns is (n*m + m*p + p*n) * rank.
    # The number of equations is n * p * m * m = n * p * m^2.
    # For n=2, m=3, p=8:
    # Unknowns: (6 + 24 + 16) * rank = 46 * rank.
    # Equations: 2 * 8 * 9 = 144.
    
    # If we assume rank = 40, unknowns = 1840, equations = 144. This is underdetermined.
    # We can try to find a solution by setting some variables to 0 and solving for the rest.
    # Or we can use a random initialization and gradient descent, but that's complex.
    
    # A simpler approach: try to construct a decomposition with rank < 48 by using known results.
    # For 2x3x8, the schoolbook rank is 48.
    # It is known that the rank of matrix multiplication can be reduced using Strassen's algorithm.
    # Strassen's algorithm for 2x2x2 has rank 7 instead of 8.
    # We can try to compose Strassen's algorithm with the 8 dimension.
    # However, 8 is not a power of 2, so we might need to pad or use a different approach.
    
    # Let's try a random search for a rank-40 decomposition.
    # We will initialize U, V, W with random values and then try to solve the linear system.
    # Since the system is underdetermined, we can fix some variables and solve for the others.
    
    # Let's try rank = 40.
    rank = 40
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We will fix the first 144 variables (equations) to 1 and solve for the rest.
    # This is a heuristic and may not work.
    
    # Instead, let's return the schoolbook decomposition for now, as it is guaranteed to be correct.
    # Future work: implement a solver for the linear system.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
