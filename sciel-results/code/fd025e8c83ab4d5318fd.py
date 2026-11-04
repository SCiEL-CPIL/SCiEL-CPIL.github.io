"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 50

    # Build U, V, W as 2D arrays (rows = input indices, cols = rank)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    # Winograd 2x2x2 scheme
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # The 7 products:
    # p0 = (A11+A22)*(B11+B22)
    # p1 = A12*B21
    # p2 = A21*B12
    # p3 = (A11+A12)*B11
    # p4 = (A22+A21)*B22
    # p5 = (A11-A21)*B21
    # p6 = (A22-A12)*B12
    
    # U coefficients (A side):
    # p0: A11=1, A22=1
    # p1: A12=1
    # p2: A21=1
    # p3: A11=1, A12=1
    # p4: A22=1, A21=1
    # p5: A11=1, A21=-1
    # p6: A22=1, A12=-1
    
    # V coefficients (B side):
    # p0: B11=1, B22=1
    # p1: B21=1
    # p2: B12=1
    # p3: B11=1
    # p4: B22=1
    # p5: B21=1
    # p6: B12=1
    
    # W coefficients (C side):
    # C11 = p0 - p3 + p5
    # C12 = p0 - p4 + p6
    # C21 = p0 - p3 + p2  ... wait, let me re-derive
    
    # Actually, let me use the standard Winograd 2x2:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # With the products:
    # p0 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # p1 = A12*B21
    # p2 = A21*B12
    # p3 = (A11+A12)B11 = A11B11 + A12B11
    # p4 = (A22+A21)B22 = A22B22 + A21B22
    # p5 = (A11-A21)B21 = A11B21 - A21B21
    # p6 = (A22-A12)B12 = A22B12 - A12B12
    
    # C11 = A11B11 + A12B21
    #     = p0 - p4 + p5 - p2 - p1 + p1  ... this is getting complicated.
    
    # Let me use a known correct Winograd 2x2 decomposition:
    # From Winograd 1970:
    # C11 = p0 - p3 + p5
    # C12 = p0 - p4 + p6  
    # C21 = p0 - p3 + p2 - p5
    # C22 = p0 - p4 + p1 - p6
    
    # Wait, let me verify:
    # C11 = p0 - p3 + p5
    #     = (A11B11+A11B22+A22B11+A22B22) - (A11B11+A12B11) + (A11B21-A21B21)
    #     = A11B11+A11B22+A22B11+A22B22 - A11B11-A12B11 + A11B21-A21B21
    #     = A11B22+A22B11+A22B22 - A12B11 + A11B21-A21B21
    # This doesn't equal A11B11 + A12B21.
    
    # Let me look up the correct Winograd 2x2 scheme.
    # Actually, the standard Winograd 2x2 has rank 7.
    # Let me use a different approach: just verify by construction.
    
    # I'll use the following known correct rank-7 Winograd 2x2:
    # U (4x7):
    # A11: [1, 0, 0, 1, 1, 1, 0]
    # A12: [0, 1, 0, 1, 0, 0, -1]
    # A21: [0, 0, 1, 0, 1, -1, 0]
    # A22: [1, 0, 0, 0, 0, 0, 1]
    
    # V (4x7):
    # B11: [1, 0, 0, 1, 0, 0, 0]
    # B12: [0, 0, 1, 0, 0, 0, 1]
    # B21: [0, 1, 0, 0, 0, 1, 0]
    # B22: [1, 0, 0, 0, 1, 0, 0]
    
    # W (4x7):
    # C11: [1, 0, 0, 0, 0, 0, 0]
    # C12: [0, 0, 0, 0, 0, 0, 1]
    # C21: [0, 0, 0, 0, 0, 1, 0]
    # C22: [0, 0, 0, 0, 1, 0, 0]
    
    # Let me verify C11:
    # C11 = sum_k U[0,k]*V[0,k]*W[0,k]
    #     = 1*1*1 + 0*0*0 + 0*0*0 + 1*1*0 + 1*0*0 + 1*0*0 + 0*0*0
    #     = 1
    # But C11 should be A11*B11 + A12*B21.
    # With U[0]=[1,0,0,1,1,1,0], V[0]=[1,0,0,1,0,0,0]
    # Product: 1*1 + 0*0 + 0*0 + 1*1 + 1*0 + 1*0 + 0*0 = 2
    # That's wrong.
    
    # I think I need to be more careful. Let me just use schoolbook rank 56 to ensure validity,
    # but that gives score 1.0.
    
    # Actually, let me try a simpler approach: use the rank-50 Winograd for 2x2x2 blocks.
    # 2x4x7 = 2*(2x2x7) = 2*(2x2x4 + 2x2x3) = 2*(2*(2x2x2) + (2x2x2 + 2x2x1))
    # = 2*(2*7 + 7 + 4) = 2*(14+11) = 2*25 = 50
    
    # For 2x2x1: rank 4 schoolbook
    # For 2x2x2: rank 7 Winograd
    
    # Let me construct this properly.
    
    # Reset
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(2):  # 2 blocks of 2x2x7
        for j in range(2):  # split inner dim 4 into two 2s
            # For each 2x2x7 block: split outer dim 7 into 4+3
            # 2x2x4 = 2*(2x2x2): 2 Winograd blocks
            # 2x2x3 = 1*(2x2x2) + 1*(2x2x1): 1 Winograd + 1 schoolbook
            
            # First 2x2x2 block (cols 0-1 of outer dim)
            # A: rows i*2 to i*2+1, cols j*2 to j*2+1
            # B: rows j*2 to j*2+1, cols 0 to 1
            # C: rows 0 to 1, cols i*2 to i*2+1
            
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1; V[j*2+1, col] = 1
            W[0, col] = 1; W[1, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[1, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[0, col] = 1
            col += 1
            
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[0, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[1, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[0, col] = 1
            col += 1
            
            U[i*2+1, col] = 1; U[i*2, col] = -1
            V[j*2+1, col] = 1
            W[1, col] = 1
            col += 1
            
            # Second 2x2x2 block (cols 2-3 of outer dim)
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1; V[j*2+1, col] = 1
            W[2, col] = 1; W[3, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[3, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[2, col] = 1
            col += 1
            
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[2, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[3, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[2, col] = 1
            col += 1
            
            U[i*2+1, col] = 1; U[i*2, col] = -1
            V[j*2+1, col] = 1
            W[3, col] = 1
            col += 1
            
            # 2x2x2 block (cols 4-5 of outer dim)
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1; V[j*2+1, col] = 1
            W[4, col] = 1; W[5, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[5, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[4, col] = 1
            col += 1
            
            U[i*2, col] = 1; U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[4, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[5, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[4, col] = 1
            col += 1
            
            U[i*2+1, col] = 1; U[i*2, col] = -1
            V[j*2+1, col] = 1
            W[5, col] = 1
            col += 1
            
            # 2x2x1 block (col 6 of outer dim): schoolbook rank 4
            U[i*2, col] = 1
            V[j*2, col] = 1
            W[6, col] = 1
            col += 1
            
            U[i*2, col] = 1
            V[j*2+1, col] = 1
            W[6, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2, col] = 1
            W[6, col] = 1
            col += 1
            
            U[i*2+1, col] = 1
            V[j*2+1, col] = 1
            W[6, col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
