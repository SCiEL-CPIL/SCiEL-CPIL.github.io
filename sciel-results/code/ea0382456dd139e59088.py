"""Matrix multiplication 2x4x5 tensor decomposition, rank 36."""
import numpy as np

DIMENSIONS = [2, 4, 5]

def _strassen_2x2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7).
    U[i*2+j,c]=coef for A[i,j], V[j*2+k,c]=coef for B[j,k], W[k*2+i,c]=coef for C[i,k].
    """
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1=(a00+a11)(b00+b11): U[0,0]=U[3,0]=1, V[0,0]=V[3,0]=1, W[0,0]=W[3,0]=1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(a10+a11)b00: U[2,1]=U[3,1]=1, V[0,1]=1, W[1,1]=W[2,1]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[2,1]=1
    # M3=a00(b10-b11): U[0,2]=1, V[2,2]=1,V[3,2]=-1, W[1,2]=1,W[3,2]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
    # M4=a11(b01-b00): U[3,3]=1, V[1,3]=1,V[0,3]=-1, W[0,3]=-1,W[2,3]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
    # M5=(a00+a10)b11: U[0,4]=U[2,4]=1, V[3,4]=1, W[2,4]=W[3,4]=1
    U[0,4]=1; U[2,4]=1; V[3,4]=1; W[2,4]=1; W[3,4]=1
    # M6=(a10-a00)(b00+b10): U[2,5]=1,U[0,5]=-1, V[0,5]=V[2,5]=1, W[0,5]=-1,W[2,5]=-1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[0,5]=-1; W[2,5]=-1
    # M7=(a01-a11)(b01+b11): U[1,6]=1,U[3,6]=-1, V[1,6]=V[3,6]=1, W[0,6]=-1,W[1,6]=1
    U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=-1; W[1,6]=1
    return U, V, W

def _schoolbook_2x2x1():
    """Schoolbook 2x2x1: U(4,4), V(2,4), W(2,4)."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # a00*b0->c0
    U[2,1]=1; V[1,1]=1; W[0,1]=1  # a10*b1->c0
    U[1,2]=1; V[0,2]=1; W[1,2]=1  # a01*b0->c1
    U[3,3]=1; V[1,3]=1; W[1,3]=1  # a11*b1->c1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    for block in range(2):  # Split m=4 into two 2-blocks
        for sub in range(3):  # Split p=5 into 2+2+1
            if sub < 2:
                sU, sV, sW = _strassen_2x2x2()
                for c in range(7):
                    for i in range(2):
                        for jl in range(2):
                            v = sU[i*2+jl, c]
                            if v: U[i*m + 2*block+jl, col+c] = v
                    for jl in range(2):
                        for kl in range(2):
                            v = sV[jl*2+kl, c]
                            if v: V[(2*block+jl)*p + 2*sub+kl, col+c] = v
                    for kl in range(2):
                        for i in range(2):
                            v = sW[kl*2+i, c]
                            if v: W[(2*sub+kl)*n + i, col+c] = v
                col += 7
            else:
                sU, sV, sW = _schoolbook_2x2x1()
                for c in range(4):
                    for i in range(2):
                        for jl in range(2):
                            v = sU[i*2+jl, c]
                            if v: U[i*m + 2*block+jl, col+c] = v
                    for jl in range(2):
                        v = sV[jl, c]
                        if v: V[(2*block+jl)*p + 4, col+c] = v
                    for i in range(2):
                        v = sW[i, c]
                        if v: W[4*n + i, col+c] = v
                col += 4
    # Convert to list of lists with clean values
    def to_list(M):
        return [[int(x) if x == int(x) else x for x in row] for row in M]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
