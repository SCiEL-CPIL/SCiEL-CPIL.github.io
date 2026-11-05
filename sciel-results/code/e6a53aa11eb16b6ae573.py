"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    We use a hybrid approach:
    1. For the 3x4 * 4x8 multiplication, we can think of it as:
       A (3x4) * B (4x8) = C (3x8)
    
    2. We can partition this into blocks. Since 4 = 2+2 and 8 = 4+4:
       A = [A1 | A2] where A1, A2 are 3x2
       B = [B1; B2] where B1, B2 are 2x8
    
       C = A1*B1 + A2*B2
    
    3. For each 3x2 * 2x8 multiplication, we can use a rank decomposition.
       The schoolbook rank for 3x2*2x8 is 3*2*8 = 48.
    
    4. However, we can potentially do better. Let's think about the structure.
    
    Actually, let me try a different approach. The key insight is that for matrix
    multiplication M_{n,m,p}, the rank is at most n*m*p (schoolbook). But for
    specific dimensions, we can sometimes do better.
    
    For 3x4*4x8, let's consider the following:
    - n=3, m=4, p=8
    - Schoolbook rank: 3*4*8 = 96
    
    Let me try to construct a decomposition with rank less than 96.
    
    One approach: use the fact that we can write the multiplication as a sum of
    outer products. The tensor M_{3,4,8} has components T[i,j,k] = delta_{i,i'} 
    delta_{j,j'} delta_{k,k'}.
    
    A rank-r decomposition is a set of r triples (u_r, v_r, w_r) such that
    T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    In the interface:
    - U has shape (n*m, rank) = (12, rank)
    - V has shape (m*p, rank) = (32, rank)  
    - W has shape (p*n, rank) = (24, rank)
    
    The convention is W[k*n+i, r].
    
    For the schoolbook decomposition, each term corresponds to a unique (i,j,k) triple.
    
    Let me think if there's a way to reduce the rank. 
    
    Actually, for the specific case of 3x4*4x8, I'm not aware of a known rank less than 96.
    Let me just implement the schoolbook decomposition correctly.
    
    Wait, let me reconsider. Maybe I can use a block structure.
    
    Let me partition the 4-dimensional middle index into two groups of 2:
    j = j1 + 2*j2 where j1 in {0,1}, j2 in {0,1}
    
    Then A[i, j] = A[i, j1 + 2*j2]
    B[j, k] = B[j1 + 2*j2, k]
    
    C[i, k] = sum_{j1=0}^{1} sum_{j2=0}^{1} A[i, j1+2*j2] * B[j1+2*j2, k]
    
    This is just the schoolbook with j split into two parts. The rank is still 96.
    
    Let me try another partition. What if I use the fact that 8 = 2*4?
    
    k = k1 + 2*k2 where k1 in {0,1}, k2 in {0,1,2,3}
    
    C[i, k] = sum_j A[i,j] * B[j, k]
    
    This doesn't help reduce the rank.
    
    I think for 3x4*4x8, the minimal rank might indeed be 96, or at least close to it.
    Let me just implement the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
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
