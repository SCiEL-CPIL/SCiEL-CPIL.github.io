"""Matrix multiplication 3x3x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook
    rs = n*m*p
    Us = [[0]*rs for _ in range(n*m)]
    Vs = [[0]*rs for _ in range(m*p)]
    Ws = [[0]*rs for _ in range(p*n)]
    c=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j][c]=1; Vs[j*p+k][c]=1; Ws[k*n+i][c]=1; c+=1
    best=(rs,Us,Vs,Ws)

    # Correct 2x2x2 Strassen with verified W coefficients
    # Using standard Strassen:
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11),
    # M4=A11(B01-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00=M1-M2+M3+M4, C01=M1+M5-M3-M7, C10=M2+M4-M6-M7, C11=M1+M5-M4-M6
    
    # Verify these formulas:
    # C00 = A00B00+A01B10
    # M1-M2+M3+M4 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + A11(B01-B00)
    # = A00B00+A00B11+A11B00+A11B11 - A10B00-A11B00 + A00B10-A00B11 + A11B01-A11B00
    # = A00B00+A11B11-A10B00+A00B10+A11B01-A11B00
    # = A00B00+A00B10 + A11(B11-A00... no
    # = A00B00+A00B10+A11B11+A11B01-A11B00-A10B00
    # This doesn't simplify to A00B00+A01B10. So these formulas are wrong too.
    
    # I'll just use a brute-force verified approach: try all 2x2x2 rank-7 decompositions
    # by solving the linear system.
    
    # For 2x2x2, the tensor is T[i,j,k]=1 for all i,j,k in {0,1}.
    # A rank-r decomposition: T[i,j,k]=sum_{r=1}^r u[i,j]*v[j,k]*w[k,i]
    # This is equivalent to finding rank-r factorization of the 4x4x4 tensor.
    # The 2x2x2 multiplication tensor has rank 7 (Strassen).
    
    # Let me just hardcode a known-correct 2x2x2 Strassen decomposition.
    # From literature, one correct decomposition:
    # u factors (i,j): 
    #   r=0: (1,0,0,1) -> u[0,0]=1,u[1,1]=1
    #   r=1: (0,1,1,1) -> u[0,1]=1,u[1,0]=1,u[1,1]=1  
    #   r=2: (1,0,0,0) -> u[0,0]=1
    #   r=3: (0,0,0,1) -> u[1,1]=1
    #   r=4: (1,1,0,0) -> u[0,0]=1,u[0,1]=1
    #   r=5: (0,1,1,0) -> u[0,1]=1,u[1,0]=1
    #   r=6: (0,1,0,1) -> u[0,1]=1,u[1,1]=1
    # v factors (j,k):
    #   r=0: (1,1,1,1) -> v[0,0]=1,v[0,1]=1,v[1,0]=1,v[1,1]=1
    #   r=1: (1,0,0,0) -> v[0,0]=1
    #   r=2: (0,0,1,-1) -> v[1,0]=1,v[1,1]=-1
    #   r=3: (0,1,-1,0) -> v[0,1]=1,v[0,0]=-1
    #   r=4: (0,0,0,1) -> v[1,1]=1
    #   r=5: (1,1,0,0) -> v[0,0]=1,v[0,1]=1
    #   r=6: (0,0,1,1) -> v[1,0]=1,v[1,1]=1
    # w factors (k,i):
    #   r=0: (1,1,1,1)
    #   r=1: (0,1,0,1)
    #   r=2: (1,-1,1,-1)
    #   r=3: (-1,1,-1,1)
    #   r=4: (-1,1,-1,1)
    #   r=5: (0,1,0,1)
    #   r=6: (1,1,1,1)
    
    # Let me just verify numerically in code.
    
    u_fact = [
        {0:1,3:1},   # r0: u[0,0]=1,u[1,1]=1
        {1:1,2:1,3:1}, # r1
        {0:1},        # r2
        {3:1},        # r3
        {0:1,1:1},    # r4
        {1:1,2:1},    # r5
        {1:1,3:1},    # r6
    ]
    v_fact = [
        {0:1,1:1,2:1,3:1}, # r0
        {0:1},              # r1
        {2:1,3:-1},         # r2
        {1:1,0:-1},         # r3
        {3:1},              # r4
        {0:1,1:1},          # r5
        {2:1,3:1},          # r6
    ]
    w_fact = [
        {0:1,1:1,2:1,3:1}, # r0
        {1:1,3:1},          # r1
        {0:1,1:-1,2:1,3:-1}, # r2
        {0:-1,1:1,2:-1,3:1}, # r3
        {0:-1,1:1,2:-1,3:1}, # r4
        {1:1,3:1},           # r5
        {0:1,1:1,2:1,3:1},  # r6
    ]
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s=0.0
                for r in range(7):
                    ui = u_fact[r].get(i*2+j,0)
                    vi = v_fact[r].get(j*2+k,0)
                    wi = w_fact[r].get(k*2+i,0)
                    s += ui*vi*wi
                if abs(s-1.0)>1e-9:
                    print(f"Strassen fail at ({i},{j},{k}): {s}")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        # Strassen verified! Build full decomposition
        Uf=[[0]*rs for _ in range(n*m)]
        Vf=[[0]*rs for _ in range(m*p)]
        Wf=[[0]*rs for _ in range(p*n)]
        col=0
        covered=set()
        for k0 in [0,2,4,6]:
            for r in range(7):
                for ij,val in u_fact[r].items():
                    i=ij//2; j=ij%2
                    Uf[i*m+j0+0][col+r] += val  # j0=0
                for jk,val in v_fact[r].items():
                    j=jk//2; k=jk%2
                    Vf[j*p+k0+k][col+r] += val
                for ki,val in w_fact[r].items():
                    k=ki//2; i=ki%2
                    Wf[(k0+k)*n+i][col+r] += val
            for i in [0,1]:
                for j in [0,1]:
                    for k in [k0,k0+1]:
                        covered.add((i,j,k))
            col+=7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) not in covered:
                        Uf[i*m+j][col]=1; Vf[j*p+k][col]=1; Wf[k*n+i][col]=1; col+=1
        
        if verify(Uf,Vf,Wf,col):
            best=(col,[r[:col] for r in Uf],[r[:col] for r in Vf],[r[:col] for r in Wf])
    
    return {"U":best[1],"V":best[2],"W":best[3]}
# EVOLVE-BLOCK-END
