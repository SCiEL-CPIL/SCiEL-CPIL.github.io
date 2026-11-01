"""Strassen rank-7 decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm gives a rank-7 decomposition for 2x2 matrix multiplication.
    # We need to map this to the interface:
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # where n=m=p=2, so:
    # U[2*i+j, c], V[2*j+k, c], W[2*k+i, c]
    #
    # Strassen's 7 products for A (2x2) * B (2x2):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Result C = [[M1+M4-M5+M7, M3+M5],
    #             [M2+M4,         M1-M2+M3+M6]]
    #
    # Now map to the tensor decomposition format.
    # The multiplication tensor T[i,j,k] = 1 if the product entry C[i,k] depends on A[i,j]*B[j,k].
    # 
    # For the rank-r decomposition:
    # T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # Wait, let me re-read the convention. The task says:
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    #
    # So the decomposition is: for each r, we have a rank-1 tensor
    # U_r (4-vector indexed by i*m+j), V_r (4-vector indexed by j*p+k), W_r (4-vector indexed by k*n+i)
    # and the tensor T[i,j,k] = sum_r U_r[2i+j] * V_r[2j+k] * W_r[2k+i]
    #
    # Let me define the 7 Strassen terms and extract U, V, W entries.
    #
    # Term 1: (a+d)(e+h)
    #   a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
    #   U[0,0] = 1 (a), U[1,1] = 1 (d) -> U indices: 2*0+0=0, 2*1+1=3
    #   V[0,0] = 1 (e), V[1,1] = 1 (h) -> V indices: 2*0+0=0, 2*1+1=3
    #   W[0,0] = 1, W[0,1] = 1, W[1,0] = 1, W[1,1] = 1
    #   Wait, W[k*n+i] for k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    #   C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1
    #   So W[0]=1, W[1]=1, W[2]=1, W[3]=1
    #
    # Let me be more careful. C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = a*e + b*g
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = a*f + b*h
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = c*e + d*g
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = c*f + d*h
    #
    # Strassen:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    #
    # C[0,0] = M1+M4-M5+M7 = (ae+ah+de+dh)+(dg-de)-(ah+bh)+(bg+bh-dg-dh)
    #         = ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh = ae+bg ✓
    # C[0,1] = M3+M5 = (af-ah)+(ah+bh) = af+bh ✓
    # C[1,0] = M2+M4 = (ce+de)+(dg-de) = ce+dg ✓
    # C[1,1] = M1-M2+M3+M6 = (ae+ah+de+dh)-(ce+de)+(af-ah)+(ce+cf-ae-af)
    #         = ae+ah+de+dh-ce-de+af-ah+ce+cf-ae-af = dh+cf ✓
    #
    # Now build U, V, W for each term r:
    #
    # r=0 (M1=(a+d)(e+h)):
    #   U: a at (0,0)->idx0, d at (1,1)->idx3 => U[0]=1, U[3]=1
    #   V: e at (0,0)->idx0, h at (1,1)->idx3 => V[0]=1, V[3]=1
    #   W: contributes to C[0,0],C[0,1],C[1,0],C[1,1] with +1 => W[0]=1,W[1]=1,W[2]=1,W[3]=1
    #
    # r=1 (M2=(c+d)e):
    #   U: c at (1,0)->idx2, d at (1,1)->idx3 => U[2]=1, U[3]=1
    #   V: e at (0,0)->idx0 => V[0]=1
    #   W: contributes to C[1,0] with +1 => W[2]=1
    #
    # r=2 (M3=a(f-h)):
    #   U: a at (0,0)->idx0 => U[0]=1
    #   V: f at (0,1)->idx1, -h at (1,1)->idx3 => V[1]=1, V[3]=-1
    #   W: contributes to C[0,1] with +1 => W[1]=1
    #
    # r=3 (M4=d(g-e)):
    #   U: d at (1,1)->idx3 => U[3]=1
    #   V: g at (1,0)->idx2, -e at (0,0)->idx0 => V[2]=1, V[0]=-1
    #   W: contributes to C[0,0] with +1, C[1,0] with +1 => W[0]=1, W[2]=1
    #
    # r=4 (M5=(a+b)h):
    #   U: a at (0,0)->idx0, b at (0,1)->idx1 => U[0]=1, U[1]=1
    #   V: h at (1,1)->idx3 => V[3]=1
    #   W: contributes to C[0,0] with -1, C[0,1] with +1 => W[0]=-1, W[1]=1
    #
    # r=5 (M6=(c-a)(e+f)):
    #   U: c at (1,0)->idx2, -a at (0,0)->idx0 => U[2]=1, U[0]=-1
    #   V: e at (0,0)->idx0, f at (0,1)->idx1 => V[0]=1, V[1]=1
    #   W: contributes to C[1,1] with +1 => W[3]=1
    #
    # r=6 (M7=(b-d)(g+h)):
    #   U: b at (0,1)->idx1, -d at (1,1)->idx3 => U[1]=1, U[3]=-1
    #   V: g at (1,0)->idx2, h at (1,1)->idx3 => V[2]=1, V[3]=1
    #   W: contributes to C[0,0] with +1 => W[0]=1
    #
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the matrices
    # r=0: M1=(a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # r=1: M2=(c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # r=2: M3=a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # r=3: M4=d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # r=4: M5=(a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # r=5: M6=(c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: M7=(b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
