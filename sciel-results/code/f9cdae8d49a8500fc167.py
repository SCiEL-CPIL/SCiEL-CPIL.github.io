"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    
    For matrices A (2x2) and B (2x2), we compute C = A * B using 7 multiplications.
    
    Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]].
    
    Strassen's formulas:
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)
    
    Then:
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    The decomposition uses the trace convention W[k*n+i, r] for the output factor.
    
    Factor matrices:
    U: shape (n*m, rank) = (4, 7)
    V: shape (m*p, rank) = (4, 7)
    W: shape (p*n, rank) = (4, 7)
    
    The tensor T satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We need to find U, V, W with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # U[i*m + j, r] for i in {0,1}, j in {0,1}
    # V[j*p + k, r] for j in {0,1}, k in {0,1}
    # W[k*n + i, r] for k in {0,1}, i in {0,1}
    
    # Let's define the 7 multiplications explicitly.
    # Each multiplication corresponds to a rank-1 term.
    
    # We'll use the standard Strassen decomposition.
    # The key is to express each entry of C as a linear combination of the M's.
    
    # Let's build U, V, W such that for each rank-1 component r:
    # U[:, r] * V[:, r] * W[:, r] contributes to the tensor.
    
    # Actually, the standard way to write Strassen as a tensor decomposition:
    # We need to find vectors u_r, v_r, w_r in the input/output spaces such that
    # sum_r u_r ⊗ v_r ⊗ w_r = matmul_tensor
    
    # For 2x2 matmul, the tensor has entries T[i,j,k] where i is row of A, j is col of A 
    # (which becomes row of B), k is col of B.
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Actually, the matmul tensor is: T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
    # No, that's not right. The tensor for matmul is:
    # T_{ijk} = 1 if the multiplication A_{ij} * B_{jk} contributes to C_{ik}
    # So T_{ijk} = 1 for all valid i,j,k.
    
    # Wait, I need to be more careful. The tensor T satisfies:
    # C_{ik} = sum_j A_{ij} B_{jk} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i, j, k.
    
    # For n=m=p=2, T is a 2x2x2 tensor with all entries = 1.
    
    # Strassen's decomposition expresses this as a sum of 7 rank-1 tensors.
    
    # Let me use the known Strassen factorization.
    
    # The standard Strassen decomposition can be written as:
    # Using the bilinear forms, we have 7 products.
    
    # Let me define the factors directly.
    
    # U has rows indexed by (i,j) where i is row of A, j is col of A
    # V has rows indexed by (j,k) where j is row of B, k is col of B  
    # W has rows indexed by (k,i) where k is col of B, i is row of C (output)
    
    # Using the trace convention W[k*n+i, r], so W index is k*n + i
    
    # Let's use the explicit Strassen factors.
    
    # I'll use a known formulation. Let me define the 7 rank-1 components.
    
    # Component 1: (a+d)*(e+h)
    # U row for a: index 0*2+0=0, V row for e: 0*2+0=0, W row for C[0,0]: 0*2+0=0
    # U row for d: index 1*2+1=3, V row for h: 1*2+1=3, W row for C[1,1]: 1*2+1=3
    
    # This is getting complex. Let me use a different approach.
    
    # I'll directly specify the factor matrices based on known Strassen implementations.
    
    # Using the formulation where:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # And:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each M_r is a product of a linear form in A and a linear form in B.
    # For M1: linear form in A is a+d, which has coefficients U[a]=1, U[d]=1, others 0
    #   In A indices: a=A[0,0] -> index 0*2+0=0, d=A[1,1] -> index 1*2+1=3
    #   So U[:, 0] = [1, 0, 0, 1] for M1 (r=0)
    #   Linear form in B is e+h, where e=B[0,0]->index 0*2+0=0, h=B[1,1]->index 1*2+1=3
    #   So V[:, 0] = [1, 0, 0, 1] for M1
    
    # For the output, M1 contributes to C00 and C11.
    # C00 has index k*n+i = 0*2+0 = 0
    # C11 has index k*n+i = 1*2+1 = 3
    # So W[:, 0] = [1, 0, 0, 1] for M1 (contributing +1 to C00 and +1 to C11)
    
    # Let me build all 7 components.
    
    # Index mapping:
    # A[i,j] -> U index i*m+j = i*2+j
    #   A[0,0]=a -> 0, A[0,1]=b -> 1, A[1,0]=c -> 2, A[1,1]=d -> 3
    # B[j,k] -> V index j*p+k = j*2+k
    #   B[0,0]=e -> 0, B[0,1]=f -> 1, B[1,0]=g -> 2, B[1,1]=h -> 3
    # C[i,k] -> W index k*n+i = k*2+i
    #   C[0,0] -> 0*2+0=0, C[0,1] -> 1*2+0=2, C[1,0] -> 0*2+1=1, C[1,1] -> 1*2+1=3
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # M1 = (a+d)(e+h), contributes +1 to C00 and +1 to C11
    # U: a(0)=1, d(3)=1
    U[0][0] = 1; U[3][0] = 1
    # V: e(0)=1, h(3)=1
    V[0][0] = 1; V[3][0] = 1
    # W: C00(0)=1, C11(3)=1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e, contributes +1 to C10
    # U: c(2)=1, d(3)=1
    U[2][1] = 1; U[3][1] = 1
    # V: e(0)=1
    V[0][1] = 1
    # W: C10(1)=1
    W[1][1] = 1
    
    # M3 = a(f-h), contributes +1 to C01 and +1 to C11
    # U: a(0)=1
    U[0][2] = 1
    # V: f(1)=1, h(3)=-1
    V[1][2] = 1; V[3][2] = -1
    # W: C01(2)=1, C11(3)=1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e), contributes +1 to C00 and +1 to C10
    # U: d(3)=1
    U[3][3] = 1
    # V: g(2)=1, e(0)=-1
    V[2][3] = 1; V[0][3] = -1
    # W: C00(0)=1, C10(1)=1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h, contributes -1 to C00 and +1 to C01
    # U: a(0)=1, b(1)=1
    U[0][4] = 1; U[1][4] = 1
    # V: h(3)=1
    V[3][4] = 1
    # W: C00(0)=-1, C01(2)=1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f), contributes +1 to C11
    # U: c(2)=1, a(0)=-1
    U[2][5] = 1; U[0][5] = -1
    # V: e(0)=1, f(1)=1
    V[0][5] = 1; V[1][5] = 1
    # W: C11(3)=1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h), contributes +1 to C00
    # U: b(1)=1, d(3)=-1
    U[1][6] = 1; U[3][6] = -1
    # V: g(2)=1, h(3)=1
    V[2][6] = 1; V[3][6] = 1
    # W: C00(0)=1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
