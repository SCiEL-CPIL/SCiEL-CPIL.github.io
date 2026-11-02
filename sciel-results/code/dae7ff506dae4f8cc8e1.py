import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS  # 2,3,4
# U: (n*m, r) = (6, r); V: (m*p, r) = (12, r); W: (p*n, r) = (8, r)

def build_T():
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def verify(U, V, W):
    # exact check using Fraction
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # build T exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(r):
                    val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if val != 1:
                    return False
    # check zeros: for all (a,b,c) not of the form, should be 0
    # T[a,b,d] = sum_c U[a,c]V[b,c]W[d,c]; for valid (i,j),(j',k),(i',k') with i=i',j=j',k=k' -> 1 else 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if not (i==i2 and j==j2 and k==k2):
                                val = Fraction(0)
                                for c in range(r):
                                    val += Fraction(U[i*m+j][c]) * Fraction(j2*p+k and V[j2*p+k][c]) * Fraction(W[k2*n+i2][c])
                                if val != 0:
                                    return False
    return True
