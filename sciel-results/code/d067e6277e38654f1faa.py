import sys
import time
import random
from fractions import Fraction

def _halfint(x):
    """Check if x is a half-integer (multiple of 0.5)."""
    return (2 * x) % 1 == 0

def build_schoolbook():
    """Build the schoolbook decomposition for 2x3 * 3x6."""
    m, n, p = 3, 6, 2
    U, V, W = [], [], []
    for i in range(p):
        for j in range(m):
            for k in range(n):
                # A is p x m, B is m x n, C is p x n
                # C[i][k] = sum_j A[i][j] * B[j][k]
                u = [0]*p
                v = [0]*m
                w = [0]*n
                u[i] = 1
                v[j] = 1
                w[k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def verify(U, V, W, p, m, n):
    """Verify that the decomposition is valid."""
    ok = True
    err = 0.0
    for i in range(p):
        for k in range(n):
            target = 1.0  # In the multiplication tensor, C[i][k] = sum_j A[i][j]*B[j][k]
            # Actually, for the standard tensor, the coefficient is 1 if the indices match
            # Let's compute the actual tensor value from the decomposition
            val = 0.0
            for r in range(len(U)):
                val += U[r][i] * V[r] * W[r][k]
            # The multiplication tensor T[i,j,k] = 1 if we're looking at the right term
            # For matrix mult, T[i,j,k] = 1 (always, as it's the sum over j)
            # Wait, the tensor is T[i,k,j] where i is row of A, k is col of B, j is shared
            # T[i,k,j] = 1 for all valid i,k,j in the sum? No.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # So the tensor T has T[i,k,j] = 1
            # The decomposition is T[i,k,j] = sum_r U[r][i] * V[r][j] * W[r][k]
            # Let's check T[i,k,j] for a specific j
            pass
    # Simpler check: compute C = A*B using the decomposition vs direct
    # But we don't have A and B, we have the tensor.
    # The tensor for matrix mult is T[i,k,j] = 1
    # So we need sum_r U[r][i] * V[r][j] * W[r][k] == 1 for all i,j,k
    ok = True
    for i in range(p):
        for j in range(m):
            for k in range(n):
                val = 0.0
                for r in range(len(U)):
                    val += U[r][i] * V[r][j] * W[r][k]
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    err = max(err, abs(val - 1.0))
    return ok, err

def main():
    p, m, n = 2, 3, 6
    U, V, W = build_schoolbook()
    rank = len(U)
    ok, err = verify(U, V, W, p, m, n)
    print("rank =", rank, "valid =", ok, "max_err =", err)
    # Check half-integer constraint.
    good = all(_halfint(x) for row in U for x in row) and \
           all(_halfint(x) for row in V for x in row) and \
           all(_halfint(x) for row in W for x in row)
    print("half-integer ok =", good)

if __name__ == "__main__":
    main()
