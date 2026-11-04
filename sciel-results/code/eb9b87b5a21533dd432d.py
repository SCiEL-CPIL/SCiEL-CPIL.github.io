"""Program to find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _construct_strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-c)(e+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U factors: rows indexed by (i,j) -> i*2+j
    # V factors: rows indexed by (j,k) -> j*2+k
    # W factors: rows indexed by (k,i) -> k*2+i
    
    # M1 = (a+d)(e+h): U has a (i=0,j=0) and d (i=1,j=1); V has e (j=0,k=0) and h (j=1,k=1)
    # This is a product of sums, so we need to expand.
    # Actually, Strassen's method uses 7 scalar multiplications.
    # Let me define the 7 terms explicitly.
    
    # Term 1: (a+d) * (e+h)
    # U: a at (0,0)->idx 0, d at (1,1)->idx 3 => U[0,0]=1, U[3,0]=1
    # V: e at (0,0)->idx 0, h at (1,1)->idx 3 => V[0,0]=1, V[3,0]=1
    # W: contributes to C11 (+1), C22 (+1)
    # C11 is (i=0,k=0)->idx 0, C22 is (i=1,k=1)->idx 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (c+d) * e
    # U: c at (1,0)->idx 2, d at (1,1)->idx 3 => U[2,1]=1, U[3,1]=1
    # V: e at (0,0)->idx 0 => V[0,1]=1
    # W: contributes to C21 (+1)
    # C21 is (i=1,k=0)->idx 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3: a * (f-h)
    # U: a at (0,0)->idx 0 => U[0,2]=1
    # V: f at (0,1)->idx 1, h at (1,1)->idx 3 => V[1,2]=1, V[3,2]=-1
    # W: contributes to C12 (+1)
    # C12 is (i=0,k=1)->idx 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 4: d * (g-e)
    # U: d at (1,1)->idx 3 => U[3,3]=1
    # V: g at (1,0)->idx 2, e at (0,0)->idx 0 => V[2,3]=1, V[0,3]=-1
    # W: contributes to C11 (+1), C21 (+1)
    # C11 idx 0, C21 idx 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: (a+b) * h
    # U: a at (0,0)->idx 0, b at (0,1)->idx 1 => U[0,4]=1, U[1,4]=1
    # V: h at (1,1)->idx 3 => V[3,4]=1
    # W: contributes to C12 (+1), C11 (-1)
    # C12 idx 1, C11 idx 0
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # Term 6: (c-d) * (f+g)
    # U: c at (1,0)->idx 2, d at (1,1)->idx 3 => U[2,5]=1, U[3,5]=-1
    # V: f at (0,1)->idx 1, g at (1,0)->idx 2 => V[1,5]=1, V[2,5]=1
    # W: contributes to C22 (+1)
    # C22 idx 3
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Term 7: (b-c) * (e+h)
    # U: b at (0,1)->idx 1, c at (1,0)->idx 2 => U[1,6]=1, U[2,6]=-1
    # V: e at (0,0)->idx 0, h at (1,1)->idx 3 => V[0,6]=1, V[3,6]=1
    # W: contributes to C11 (+1)
    # C11 idx 0
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    """Construct a decomposition for 3x4 * 4x6 matrix multiplication.
    
    We use a block structure. Since 3, 4, 6 are small, we can try to build
    from known smaller decompositions or use a direct approach.
    
    For now, let's use the schoolbook method but see if we can do better.
    Actually, the task asks for rank-minimal. The schoolbook rank is 3*4*6 = 72.
    
    Let's try to use a Kronecker product approach. If we can decompose
    3x4 * 4x6 using blocks, we might reduce the rank.
    
    One approach: pad to 4x4 * 4x4 and use Strassen? No, that increases rank.
    
    Another approach: use the fact that 4 = 2*2 and 6 = 2*3.
    We can write the 3x4 matrix as a block matrix and the 4x6 matrix similarly.
    
    Let's try a different strategy: use the schoolbook decomposition and then
    apply Gaussian elimination to find a smaller rank decomposition.
    
    Actually, for the purpose of this task, let's first return the schoolbook
    decomposition to ensure correctness, and then try to improve it.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
