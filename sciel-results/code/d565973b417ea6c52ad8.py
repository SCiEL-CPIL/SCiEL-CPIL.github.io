import numpy as np

def construct():
    # 4x4 matrix times 4x8 matrix
    # n=4, m=4, p=8
    # Rank 98 via Strassen composition
    
    # Strassen 2x2x2 decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11)
    U2[0,0]+=1; U2[3,0]+=1
    V2[0,0]+=1; V2[3,0]+=1
    W2[0,0]+=1; W2[3,0]+=1
    # P2 = (A10+A11)B00
    U2[2,1]+=1; U2[3,1]+=1
    V2[0,1]+=1
    W2[2,1]+=1; W2[3,1]-=1
    # P3 = A00(B01-B11)
    U2[0,2]+=1
    V2[1,2]+=1; V2[3,2]-=1
    W2[1,2]+=1; W2[3,2]+=1
    # P4 = A11(B10-B00)
    U2[3,3]+=1
    V2[2,3]+=1; V2[0,3]-=1
    W2[0,3]+=1; W2[2,3]+=1
    # P5 = (A00+A01)B11
    U2[0,4]+=1; U2[1,4]+=1
    V2[3,4]+=1
    W2[0,4]-=1; W2[1,4]+=1
    # P6 = (A10-A00)(B00+B01)
    U2[2,5]+=1; U2[0,5]-=1
    V2[0,5]+=1; V2[1,5]+=1
    W2[3,5]+=1
    # P7 = (A01-A11)(B10+B11)
    U2[1,6]+=1; U2[3,6]-=1
    V2[2,6]+=1; V2[3,6]+=1
    W2[0,6]+=1
    
    # Build 4x4x4 decomposition (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Effective block specifications for each of 7 Strassen terms
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)]
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)]
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)]
    ]
    
    for k in range(7):
        for mm in range(7):
            col = k*7+mm
            for ca, ablk in A_specs[k]:
                bi,bj = divmod(ablk,2)
                for lr in range(2):
                    for lc in range(2):
                        gi = (bi*2+lr)*4 + (bj*2+lc)
                        li = lr*2+lc
                        U4[gi,col] += ca*U2[li,mm]
            for cb, bblk in B_specs[k]:
                bi,bj = divmod(bblk,2)
                for lr in range(2):
                    for lc in range(2):
                        gi = (bi*2+lr)*4 + (bj*2+lc)
                        li = lr*2+lc
                        V4[gi,col] += cb*V2[li,mm]
            for cc, cblk in C_specs[k]:
                bi,bj = divmod(cblk,2)
                for lr in range(2):
                    for lc in range(2):
                        # W index: col*4 + row
                        gi = (bj*2+lc)*4 + (bi*2+lr)
                        li = lr*2+lc
                        W4[gi,col] += cc*W2[li,mm]
    
    # Now compose for 4x4x8: split B into two 4x4 halves
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left half (B columns 0-3)
    U[:, :49] = U4
    for i in range(16):
        r,c = divmod(i,4)
        V[r*8+c, :49] = V4[i]
    W[:16, :49] = W4
    
    # Right half (B columns 4-7)
    U[:, 49:] = U4
    for i in range(16):
        r,c = divmod(i,4)
        V[r*8+(c+4), 49:] = V4[i]
    W[16:, 49:] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
