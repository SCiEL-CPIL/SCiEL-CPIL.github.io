"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    V[2, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1  # k=0,i=0
    W[1, 0] = 1  # k=1,i=0
    W[2, 0] = 1  # k=0,i=1
    W[3, 0] = 1  # k=1,i=1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # k=0,i=1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1  # k=1,i=0
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = -1
    W[2, 3] = 1  # k=0,i=1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # k=1,i=0
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[2, 5] = 1
    W[0, 5] = 1  # k=0,i=0
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = -1  # k=0,i=0
    
    return U, V, W

def _strassen_232():
    """Return U, V, W for 2x3x2 matmul with rank 11.
    Convention: U[i*3+j, r] (i in 0..1, j in 0..2), V[j*2+k, r] (j in 0..2, k in 0..1), W[k*2+i, r] (k in 0..1, i in 0..1)
    """
    # 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # Schoolbook: 12 terms. We use rank 11.
    # Split the 3 columns of A into 2+1, and 2 rows of B into 1+1.
    # A = [A_left (2x2) | A_right (2x1)], B = [B_top (2x2); B_bottom (1x2)]
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 2x2x2, rank 7 (Strassen)
    # A_right * B_bottom: 2x1x2, rank 4 (schoolbook)
    # Total: 11
    U = np.zeros((6, 11))  # 2*3 = 6
    V = np.zeros((6, 11))  # 3*2 = 6
    W = np.zeros((4, 11))  # 2*2 = 4
    
    sU, sV, sW = _strassen_222()
    
    col = 0
    # Part 1: A_left (2x2) * B_top (2x2), using Strassen
    for r in range(7):
        # U: A_left, i in {0,1}, j in {0,1}
        U[0*3 + 0, col] = sU[0, r]  # i=0,j=0
        U[0*3 + 1, col] = sU[1, r]  # i=0,j=1
        U[1*3 + 0, col] = sU[2, r]  # i=1,j=0
        U[1*3 + 1, col] = sU[3, r]  # i=1,j=1
        # V: B_top, j in {0,1}, k in {0,1}
        V[0*2 + 0, col] = sV[0, r]  # j=0,k=0
        V[0*2 + 1, col] = sV[1, r]  # j=0,k=1
        V[1*2 + 0, col] = sV[2, r]  # j=1,k=0
        V[1*2 + 1, col] = sV[3, r]  # j=1,k=1
        # W: C, k in {0,1}, i in {0,1}
        W[0*2 + 0, col] = sW[0, r]  # k=0,i=0
        W[0*2 + 1, col] = sW[1, r]  # k=0,i=1
        W[1*2 + 0, col] = sW[2, r]  # k=1,i=0
        W[1*2 + 1, col] = sW[3, r]  # k=1,i=1
        col += 1
    # col = 7
    
    # Part 2: A_right (2x1) * B_bottom (1x2), schoolbook rank 4
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1   # A_right: i in {0,1}, j=2
            V[2*2 + k, col] = 1   # B_bottom: j=2, k in {0,1}
            W[k*2 + i, col] = 1   # C: k, i
            col += 1
    # col = 11
    
    assert col == 11
    return U, V, W

def _verify_222():
    """Verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_222()
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(7):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _build_2x5x6():
    """
    Build 2x5x6 decomposition using block Strassen.
    Split 5 = 2+3, 6 = 2+2+2.
    
    A = [A1 (2x2) | A2 (2x3)]
    B = [B1 (2x6); B2 (3x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2x6, split B1 into three 2x2 blocks. Each uses Strassen (rank 7). Total: 21.
    A2*B2: 2x3x6, split into three 2x3x2 blocks. Each uses rank-11 decomposition. Total: 33.
    Grand total: 21 + 33 = 54
    """
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n*m, rank))  # 10 x 54
    V = np.zeros((m*p, rank))  # 30 x 54
    W = np.zeros((p*n, rank))  # 12 x 54
    
    sU, sV, sW = _strassen_222()  # 4x7 each
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A1 indices, i in {0,1}, j in {0,1} -> U[i*5+j, col]
            U[0*5 + 0, col] = sU[0, r]  # i=0,j=0
            U[0*5 + 1, col] = sU[1, r]  # i=0,j=1
            U[1*5 + 0, col] = sU[2, r]  # i=1,j=0
            U[1*5 + 1, col] = sU[3, r]  # i=1,j=1
            
            # V: B1_block indices, j in {0,1} globally, k in {k_offset, k_offset+1}
            V[0*6 + k_offset + 0, col] = sV[0, r]  # j=0,k=k_offset
            V[0*6 + k_offset + 1, col] = sV[1, r]  # j=0,k=k_offset+1
            V[1*6 + k_offset + 0, col] = sV[2, r]  # j=1,k=k_offset
            V[1*6 + k_offset + 1, col] = sV[3, r]  # j=1,k=k_offset+1
            
            # W: C indices, k in {k_offset, k_offset+1}, i in {0,1}
            W[(k_offset+0)*2 + 0, col] = sW[0, r]  # k=k_offset, i=0
            W[(k_offset+0)*2 + 1, col] = sW[1, r]  # k=k_offset, i=1
            W[(k_offset+1)*2 + 0, col] = sW[2, r]  # k=k_offset+1, i=0
            W[(k_offset+1)*2 + 1, col] = sW[3, r]  # k=k_offset+1, i=1
            
            col += 1
    # col should be 21 here
    
    # Part 2: A2 * B2 (2x3 x 6), three 2x3x2 blocks using rank-11
    s2U, s2V, s2W = _strassen_232()  # 6x11, 6x11, 4x11
    for block in range(3):
        k_offset = block * 2
        for r in range(11):
            # U: A2 indices, i in {0,1}, j in {2,3,4}
            # Local j: 0->global 2, 1->global 3, 2->global 4
            U[0*5 + 2, col] = s2U[0, r]  # i=0, local j=0 -> global j=2
            U[0*5 + 3, col] = s2U[1, r]  # i=0, local j=1 -> global j=3
            U[0*5 + 4, col] = s2U[2, r]  # i=0, local j=2 -> global j=4
            U[1*5 + 2, col] = s2U[3, r]  # i=1, local j=0 -> global j=2
            U[1*5 + 3, col] = s2U[4, r]  # i=1, local j=1 -> global j=3
            U[1*5 + 4, col] = s2U[5, r]  # i=1, local j=2 -> global j=4
            
            # V: B2_block, j in {2,3,4} globally, k in {k_offset, k_offset+1}
            # Local j: 0->global 2, 1->global 3, 2->global 4
            V[2*6 + k_offset + 0, col] = s2V[0, r]  # local j=0, k=k_offset
            V[2*6 + k_offset + 1, col] = s2V[1, r]  # local j=0, k=k_offset+1
            V[3*6 + k_offset + 0, col] = s2V[2, r]  # local j=1, k=k_offset
            V[3*6 + k_offset + 1, col] = s2V[3, r]  # local j=1, k=k_offset+1
            V[4*6 + k_offset + 0, col] = s2V[4, r]  # local j=2, k=k_offset
            V[4*6 + k_offset + 1, col] = s2V[5, r]  # local j=2, k=k_offset+1
            
            # W: C indices, k in {k_offset, k_offset+1}, i in {0,1}
            W[(k_offset+0)*2 + 0, col] = s2W[0, r]  # k=k_offset, i=0
            W[(k_offset+0)*2 + 1, col] = s2W[1, r]  # k=k_offset, i=1
            W[(k_offset+1)*2 + 0, col] = s2W[2, r]  # k=k_offset+1, i=0
            W[(k_offset+1)*2 + 1, col] = s2W[3, r]  # k=k_offset+1, i=1
            
            col += 1
    # col should be 21 + 33 = 54 here
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_full(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition."""
    rank = len(U[0])
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _check_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                if abs(val * 2 - round(val * 2)) > 1e-12:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
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

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Try Strassen-based rank-54 decomposition
    try:
        if _verify_222():
            # Verify the 2x3x2 decomposition
            U232, V232, W232 = _strassen_232()
            valid_232 = True
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        total = 0.0
                        for r in range(11):
                            total += U232[i*3+j, r] * V232[j*2+k, r] * W232[k*2+i, r]
                        if abs(total - 1.0) > 1e-10:
                            valid_232 = False
                            break
                    if not valid_232:
                        break
                if not valid_232:
                    break
            
            if valid_232:
                result = _build_2x5x6()
                if _verify_full(result['U'], result['V'], result['W']):
                    if _check_valid_entries(result['U'], result['V'], result['W']):
                        return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
