"""Matrix multiplication 4x4x6 via Kronecker product decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Build 2x2x2 Strassen factors (rank 7)
    r1 = 7
    U1 = np.zeros((4, r1), dtype=object)
    V1 = np.zeros((4, r1), dtype=object)
    W1 = np.zeros((4, r1), dtype=object)
    # M1=(a00+a11)(b00+b11)
    U1[0,0]=1; U1[3,0]=1; V1[0,0]=1; V1[3,0]=1; W1[0,0]=1; W1[3,0]=1
    # M2=(a10+a11)b00
    U1[2,1]=1; U1[3,1]=1; V1[0,1]=1; W1[1,1]=1; W1[3,1]=-1
    # M3=a00(b01-b11)
    U1[0,2]=1; V1[1,2]=1; V1[3,2]=-1; W1[2,2]=1; W1[3,2]=1
    # M4=a11(b10-b00)
    U1[3,3]=1; V1[2,3]=1; V1[0,3]=-1; W1[0,3]=1; W1[1,3]=1
    # M5=(a00+a01)b11
    U1[0,4]=1; U1[1,4]=1; V1[3,4]=1; W1[0,4]=-1; W1[2,4]=1
    # M6=(a10-a00)(b00+b01)
    U1[2,5]=1; U1[0,5]=-1; V1[0,5]=1; V1[1,5]=1; W1[3,5]=1
    # M7=(a01-a11)(b10+b11)
    U1[1,6]=1; U1[3,6]=-1; V1[2,6]=1; V1[3,6]=1; W1[0,6]=1

    # Build 2x2x3 factors (rank 11): Strassen(7) + trivial 3rd col(4)
    r2 = 11
    U2 = np.zeros((4, r2), dtype=object)
    V2 = np.zeros((6, r2), dtype=object)
    W2 = np.zeros((6, r2), dtype=object)
    # Copy Strassen into first 7 columns
    U2[:, :7] = U1
    V2[0, :7] = V1[0]; V2[1, :7] = V1[1]; V2[3, :7] = V1[2]; V2[4, :7] = V1[3]
    W2[0, :7] = W1[0]; W2[1, :7] = W1[1]; W2[2, :7] = W1[2]; W2[3, :7] = W1[3]
    # Trivial terms for 3rd column
    U2[0,7]=1; V2[2,7]=1; W2[4,7]=1
    U2[1,8]=1; V2[5,8]=1; W2[4,8]=1
    U2[2,9]=1; V2[2,9]=1; W2[5,9]=1
    U2[3,10]=1; V2[5,10]=1; W2[5,10]=1

    # Kronecker product of factor matrices
    r = r1 * r2
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
