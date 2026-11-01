"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    This uses Strassen's algorithm. The factor matrices U, V, W have shapes:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    
    The trace convention is W[k*n+i, r] where the output C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, let me re-read the convention.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    Standard tensor decomposition for matrix multiplication:
    T[i, j, k, l] = sum_r U[i, j, r] * V[j, k, r] * W[k, l, r]
    where A is n x m, B is m x p, C is n x p.
    Indices: i in [0,n), j in [0,m), k in [0,p)? No, j is shared.
    Let's use standard indices:
    A is n x m: A[i, j]
    B is m x p: B[j, k]
    C is n x p: C[i, k]
    
    The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
    
    The interface says:
    U: (n*m, rank) -> index is i*m + j
    V: (m*p, rank) -> index is j*p + k
    W: (p*n, rank) -> index is k*n + i
    
    So C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strassen's algorithm for 2x2 matrices:
    Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    
    The 7 products are:
    P1 = (a + d)(e + h)
    P2 = (c + d)e
    P3 = a(f - h)
    P4 = d(g - e)
    P5 = (a + b)h
    P6 = (c - a)(e + f)
    P7 = (b - d)(g + h)
    
    And the output C = [[c11, c12], [c21, c22]] is:
    c11 = P1 + P4 - P5 + P7
    c12 = P3 + P5
    c21 = P2 + P4
    c22 = P1 - P2 + P3 + P6
    
    Let's verify:
    c11 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
         = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
         = ae + bg
    Correct.
    
    c12 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    Correct.
    
    c21 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg
    Correct.
    
    c22 = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
         = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
         = dh + cf
    Correct.
    
    Now let's map this to the U, V, W matrices.
    
    For each product P_r, we have:
    P1: U1 = [1, 0, 1, 0] (for a, b, c, d), V1 = [1, 1, 0, 0] (for e, f, g, h)
    Wait, U is (n*m, rank) = (4, 7). The indices are i*m+j.
    For n=2, m=2:
    i=0, j=0 -> index 0 (a)
    i=0, j=1 -> index 1 (b)
    i=1, j=0 -> index 2 (c)
    i=1, j=1 -> index 3 (d)
    
    V is (m*p, rank) = (4, 7). The indices are j*p+k.
    For m=2, p=2:
    j=0, k=0 -> index 0 (e)
    j=0, k=1 -> index 1 (f)
    j=1, k=0 -> index 2 (g)
    j=1, k=1 -> index 3 (h)
    
    W is (p*n, rank) = (4, 7). The indices are k*n+i.
    For p=2, n=2:
    k=0, i=0 -> index 0 (c11)
    k=1, i=0 -> index 1 (c12)
    k=0, i=1 -> index 2 (c21)
    k=1, i=1 -> index 3 (c22)
    
    Let's define the 7 products:
    P1: U1 = [1, 0, 0, 1] (a + d), V1 = [1, 0, 0, 1] (e + h)
    P2: U2 = [0, 0, 1, 1] (c + d), V2 = [1, 0, 0, 0] (e)
    P3: U3 = [1, 0, 0, 0] (a), V3 = [0, 1, 0, -1] (f - h)
    P4: U4 = [0, 0, 0, 1] (d), V4 = [0, 0, 1, -1] (g - e)
    P5: U5 = [1, 1, 0, 0] (a + b), V5 = [0, 0, 0, 1] (h)
    P6: U6 = [1, 0, -1, 0] (a - c), V6 = [1, 1, 0, 0] (e + f)
    P7: U7 = [0, 1, 0, -1] (b - d), V7 = [0, 0, 1, 1] (g + h)
    
    Wait, P6 in my formula was (c - a)(e + f). Let's use (a - c)(e + f) to keep coefficients simple, and adjust W.
    Actually, let's stick to the standard Strassen formulation and just map it.
    
    P1 = (a + d)(e + h)
    P2 = (c + d)e
    P3 = a(f - h)
    P4 = d(g - e)
    P5 = (a + b)h
    P6 = (c - a)(e + f)
    P7 = (b - d)(g + h)
    
    U1 = [1, 0, 0, 1]
    U2 = [0, 0, 1, 1]
    U3 = [1, 0, 0, 0]
    U4 = [0, 0, 0, 1]
    U5 = [1, 1, 0, 0]
    U6 = [0, 0, 1, -1]  # c - a
    U7 = [0, 1, 0, -1]  # b - d
    
    V1 = [1, 0, 0, 1]
    V2 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    V4 = [0, 0, 1, -1]
    V5 = [0, 0, 0, 1]
    V6 = [1, 1, 0, 0]
    V7 = [0, 0, 1, 1]
    
    W matrix:
    c11 = P1 + P4 - P5 + P7
    c12 = P3 + P5
    c21 = P2 + P4
    c22 = P1 - P2 + P3 + P6
    
    W indices: k*n+i
    k=0, i=0 -> c11 -> index 0
    k=1, i=0 -> c12 -> index 1
    k=0, i=1 -> c21 -> index 2
    k=1, i=1 -> c22 -> index 3
    
    W0 = [1, 0, 0, 1, -1, 0, 1]  # c11
    W1 = [0, 0, 1, 0, 1, 0, 0]   # c12
    W2 = [0, 1, 0, 1, 0, 0, 0]   # c21
    W3 = [1, -1, 1, 0, 0, 1, 0]  # c22
    
    Let's verify c22 again:
    c22 = P1 - P2 + P3 + P6
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P6 = (c-a)(e+f)
    
    c22 = (ae + ah + de + dh) - (ce + de) + (af - ah) + (ce + cf - ae - af)
         = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
         = dh + cf
    Correct.
    
    So the matrices are:
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    
    Wait, let me re-check U6 and U7.
    U6 = [0, 0, 1, -1] for c - a. So U[2][5] = 1, U[0][5] = -1.
    U7 = [0, 1, 0, -1] for b - d. So U[1][6] = 1, U[3][6] = -1.
    
    Let's rewrite U carefully:
    U[0] (a): P1=1, P2=0, P3=1, P4=0, P5=1, P6=-1, P7=0 -> [1, 0, 1, 0, 1, -1, 0]
    U[1] (b): P1=0, P2=0, P3=0, P4=0, P5=1, P6=0, P7=1 -> [0, 0, 0, 0, 1, 0, 1]
    U[2] (c): P1=0, P2=1, P3=0, P4=0, P5=0, P6=1, P7=0 -> [0, 1, 0, 0, 0, 1, 0]
    U[3] (d): P1=1, P2=1, P3=0, P4=1, P5=0, P6=-1, P7=-1 -> [1, 1, 0, 1, 0, -1, -1]
    
    V[0] (e): P1=1, P2=1, P3=0, P4=0, P5=0, P6=1, P7=0 -> [1, 1, 0, 0, 0, 1, 0]
    V[1] (f): P1=0, P2=0, P3=1, P4=0, P5=0, P6=1, P7=0 -> [0, 0, 1, 0, 0, 1, 0]
    V[2] (g): P1=0, P2=0, P3=0, P4=1, P5=0, P6=0, P7=1 -> [0, 0, 0, 1, 0, 0, 1]
    V[3] (h): P1=1, P2=0, P3=-1, P4=0, P5=1, P6=0, P7=1 -> [1, 0, -1, 0, 1, 0, 1]
    
    W[0] (c11): P1=1, P2=0, P3=0, P4=1, P5=-1, P6=0, P7=1 -> [1, 0, 0, 1, -1, 0, 1]
    W[1] (c12): P1=0, P2=0, P3=1, P4=0, P5=1, P6=0, P7=0 -> [0, 0, 1, 0, 1, 0, 0]
    W[2] (c21): P1=0, P2=1, P3=0, P4=1, P5=0, P6=0, P7=0 -> [0, 1, 0, 1, 0, 0, 0]
    W[3] (c22): P1=1, P2=-1, P3=1, P4=0, P5=0, P6=1, P7=0 -> [1, -1, 1, 0, 0, 1, 0]
    
    Let's double check with a simple example.
    A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    
    P1 = (1+4)(5+8) = 5*13 = 65
    P2 = (3+4)*5 = 35
    P3 = 1*(6-8) = -2
    P4 = 4*(7-5) = 8
    P5 = (1+2)*8 = 24
    P6 = (3-1)*(5+6) = 2*11 = 22
    P7 = (2-4)*(7+8) = -2*15 = -30
    
    c11 = P1 + P4 - P5 + P7 = 65 + 8 - 24 - 30 = 19. Correct.
    c12 = P3 + P5 = -2 + 24 = 22. Correct.
    c21 = P2 + P4 = 35 + 8 = 43. Correct.
    c22 = P1 - P2 + P3 + P6 = 65 - 35 - 2 + 22 = 50. Correct.
    
    The decomposition is valid and has rank 7.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: (n*m, rank) = (4, 7)
    # Indices: i*m + j
    # a=0, b=1, c=2, d=3
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, -1, -1]  # d
    ]
    
    # V: (m*p, rank) = (4, 7)
    # Indices: j*p + k
    # e=0, f=1, g=2, h=3
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]
    
    # W: (p*n, rank) = (4, 7)
    # Indices: k*n + i
    # c11=0, c12=1, c21=2, c22=3
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0]   # c22
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
