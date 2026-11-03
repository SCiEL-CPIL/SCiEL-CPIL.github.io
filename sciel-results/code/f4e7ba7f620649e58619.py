"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]


def _add_term(U, V, W, u_idx, v_idx, w_idx, coeff):
    """Add a rank-1 term with given indices and coefficient."""
    col = len(W[0]) if W[0] else 0
    U[u_idx][col] = U[u_idx][col] + coeff
    V[v_idx][col] = V[v_idx][col] + coeff
    W[w_idx][col] = W[w_idx][col] + coeff


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # A (3x4): A_top (2x4), A_bot (1x4)
    # B (4x5): B_top (2x5), B_bot (2x5)
    # 
    # C_top (2x5) = A_top[:,0:2] @ B_top + A_top[:,2:4] @ B_bot
    # C_bot (1x5) = A_bot[:,0:2] @ B_top + A_bot[:,2:4] @ B_bot
    #
    # For (2x2)@(2x5): use Strassen rank-7
    # For (1x2)@(2x5): use rank-2 (dot product)
    
    # We'll build U (3*4=12 x rank), V (4*5=20 x rank), W (5*3=15 x rank)
    # U index: i*m + j = i*4 + j for A[i,j]
    # V index: j*p + k = j*5 + k for B[j,k]
    # W index: k*n + i = k*3 + i for C[k,i]
    
    # Start with empty lists, we'll determine rank dynamically
    U = []
    V = []
    W = []
    
    def new_term():
        U.append([0])
        V.append([0])
        W.append([0])
        return len(W) - 1
    
    def add_to_term(term, u_idx, v_idx, w_idx, coeff):
        U[u_idx][term] += coeff
        V[v_idx][term] += coeff
        W[w_idx][term] += coeff
    
    # Helper: add Strassen decomposition for (2x2)@(2x5)
    # a00, a01, a10, a11 are U indices for A block
    # b00..b04, b10..b14 are V indices for B block (b0k = B_top[k], b1k = B_bot[k])
    # c00..c04, c10..c14 are W indices for C block
    def strassen_2x2_2x5(a00, a01, a10, a11, b0, b1, c0, c1):
        # b0 = [b00,b01,b02,b03,b04], b1 = [b10,b11,b12,b13,b14]
        # c0 = [c00,c01,c02,c03,c04], c1 = [c10,c11,c12,c13,c14]
        
        # M1 = (A00+A11) * (B0+B1)
        t1 = new_term()
        add_to_term(t1, a00, -1, -1, 1)  # placeholder, we'll fix
        # Actually, let me restructure. Each Strassen term involves products.
        # M1_k = (A00+A11)*(B0k+B1k)
        # This is: A00*B0k + A00*B1k + A11*B0k + A11*B1k
        # But we want to express as rank-1 terms in the tensor sense.
        # The term (A00+A11)*(B0k+B1k) for all k means:
        # For each k: coefficient (A00+A11) multiplies (B0k+B1k)
        # In U: A00 gets +1, A11 gets +1
        # In V: B0k gets +1, B1k gets +1 for all k
        # In W: C0k gets +1, C1k gets... wait, no.
        
        # Let me think differently. The Strassen decomposition gives 7 products.
        # Each product is a scalar times a vector.
        # Product r has:
        #   U part: linear combination of A entries
        #   V part: linear combination of B entries (for all k)
        #   W part: linear combination of C entries (for all k)
        
        # M1 = (A00+A11) * (B0+B1)
        # U: A00:1, A11:1
        # V: B0k:1 for all k, B1k:1 for all k
        # W: C0k:1 for all k (since C0 = M1 + M4 - M5 + M7, C0k gets M1k)
        #     C1k:1 for all k (since C1 = M1 - M2 + M3 + M6, C1k gets M1k)
        
        t1 = new_term()
        U[a00][t1] += 1
        U[a11][t1] += 1
        for k in range(5):
            V[b0[k]][t1] += 1
            V[b1[k]][t1] += 1
            W[c0[k]][t1] += 1
            W[c1[k]][t1] += 1
        
        # M2 = (A00+A11) * B1
        t2 = new_term()
        U[a00][t2] += 1
        U[a11][t2] += 1
        for k in range(5):
            V[b1[k]][t2] += 1
            W[c1[k]][t2] -= 1  # C1 = M1 - M2 + M3 + M6
        
        # M3 = A01 * (B1 - B0)
        t3 = new_term()
        U[a01][t3] += 1
        for k in range(5):
            V[b1[k]][t3] += 1
            V[b0[k]][t3] -= 1
            W[c0[k]][t3] += 1  # C0 = M3 + M4 + M6... wait
            # C00 = M1 + M4 - M5 + M7, C01 = M3 + M4 + M6
            # For 2x2@2x5: C0 = M1 + M4 - M5 + M7? No.
            # Let me re-derive properly.
        
        # I think I'm making this too complicated. Let me use a different approach.
        # Just use schoolbook for each (2x2)@(2x5) block: rank = 2*2*5 = 20
        # And (1x2)@(2x5): rank = 1*2*5 = 10
        # Total: 20 + 20 + 10 + 10 = 60. Same as full schoolbook.
        
        # That's not helpful. Let me try Strassen more carefully.
        
        # For 2x2 @ 2x5, the tensor has dimensions (2, 2, 5).
        # Rank of this tensor: we can use Strassen's 7 products.
        # Each product is a scalar (from A) times a vector (from B, length 5).
        # The result contributes to C (2x5).
        
        # Let me define the 7 terms properly:
        # Term 1: U: A00:1, A11:1; V: B0k:1, B1k:1 for all k; W: C0k:1, C1k:1 for all k
        # Term 2: U: A00:1, A11:1; V: B1k:1 for all k; W: C1k:-1 for all k
        # Term 3: U: A01:1; V: B1k:1, B0k:-1 for all k; W: C0k:1 for all k
        # Term 4: U: A00:1; V: B0k:1 for all k; W: C0k:1, C1k:1 for all k
        # Term 5: U: A11:1; V: B0k:1, B1k:-1 for all k; W: C0k:-1, C1k:1 for all k
        # Term 6: U: A01:1, A10:1; V: B0k:1 for all k; W: C0k:1, C1k:1 for all k
        # Term 7: U: A10:1, A00:-1; V: B0k:1, B1k:1 for all k; W: C0k:1 for all k
        
        # Let me verify with C0k = sum_j A0j * Bjk = A00*B0k + A01*B1k
        # From terms: T1: +1, T3: +1, T4: +1, T5: -1, T6: +1, T7: +1
        # Sum for A00*B0k: T1(+1)+T4(+1) = 2? That's wrong.
        
        # I think I'm confusing myself. Let me just use the standard Strassen formulas
        # and map them directly.
        
        pass  # Will implement below properly
    
    # Let me restart with a cleaner approach.
    # I'll build the decomposition term by term.
    
    U = []
    V = []
    W = []
    
    def new_term():
        U.append([0.0])
        V.append([0.0])
        W.append([0.0])
        return len(W) - 1
    
    def add(t, u_idx, v_idx, w_idx, coeff):
        U[u_idx][t] += coeff
        V[v_idx][t] += coeff
        W[w_idx][t] += coeff
    
    # Strassen for (2x2)@(2x5):
    # A indices: a00, a01, a10, a11
    # B indices: b0k (k=0..4) for top row, b1k (k=0..4) for bottom row
    # C indices: c0k (k=0..4) for top row, c1k (k=0..4) for bottom row
    def strassen_block(a00, a01, a10, a11, b0, b1, c0, c1):
        # M1 = (A00+A11)*(B0+B1)
        t = new_term()
        add(t, a00, -1, -1, 1)  # dummy
        U[a00][t] += 1
        U[a11][t] += 1
        for k in range(5):
            V[b0[k]][t] += 1
            V[b1[k]][t] += 1
            W[c0[k]][t] += 1
            W[c1[k]][t] += 1
        
        # M2 = (A00+A11)*B1
        t = new_term()
        U[a00][t] += 1
        U[a11][t] += 1
        for k in range(5):
            V[b1[k]][t] += 1
            W[c1[k]][t] -= 1
        
        # M3 = A01*(B1-B0)
        t = new_term()
        U[a01][t] += 1
        for k in range(5):
            V[b1[k]][t] += 1
            V[b0[k]][t] -= 1
            W[c0[k]][t] += 1
        
        # M4 = A00*B0
        t = new_term()
        U[a00][t] += 1
        for k in range(5):
            V[b0[k]][t] += 1
            W[c0[k]][t] += 1
            W[c1[k]][t] += 1
        
        # M5 = A11*(B0-B1)
        t = new_term()
        U[a11][t] += 1
        for k in range(5):
            V[b0[k]][t] += 1
            V[b1[k]][t] -= 1
            W[c0[k]][t] -= 1
            W[c1[k]][t] += 1
        
        # M6 = (A01+A10)*B0
        t = new_term()
        U[a01][t] += 1
        U[a10][t] += 1
        for k in range(5):
            V[b0[k]][t] += 1
            W[c0[k]][t] += 1
            W[c1[k]][t] += 1
        
        # M7 = (A10-A00)*(B0+B1)
        t = new_term()
        U[a10][t] += 1
        U[a00][t] -= 1
        for k in range(5):
            V[b0[k]][t] += 1
            V[b1[k]][t] += 1
            W[c0[k]][t] += 1
        
        # Verify: C0k = A00*B0k + A01*B1k
        # From M1: +1, M3: +1, M4: +1, M5: -1, M6: +1, M7: +1
        # Coeff for A00*B0k: M1(1)+M4(1) = 2? No, M1 has (A00+A11)*(B0+B1) so it contributes to A00*B0k with coeff 1.
        # M4 has A00*B0k with coeff 1.
        # So A00*B0k total: 1+1 = 2. That's wrong, should be 1.
        
        # I see the issue. The Strassen formulas are for computing C, not for individual products.
        # C00 = M1 + M4 - M5 + M7, etc.
        # So the W coefficients should reflect this.
        
        # Let me redo. The W part for each term should be:
        # M1: contributes to C00 and C11 (in 2x2 case). For 2x2@2x5: C0 and C1.
        # C0 = M1 + M4 - M5 + M7 (for the "C00" position) -- but in 2x2@2x5, there's no "position", just rows.
        
        # I think the correct mapping is:
        # C0k = M1k + M4k - M5k + M7k  (this is C[0,k])
        # C1k = M2k + M4k + M5k          (this is C[1,k])
        
        # Wait, that's not standard Strassen. Standard:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M4 + M6
        # C10 = M2 + M4 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # For 2x2 @ 2x5, we have C[0,:] and C[1,:].
        # C[0,k] = A00*B0k + A01*B1k
        # C[1,k] = A10*B0k + A11*B1k
        
        # Using Strassen:
        # C[0,k] = M1k + M4k - M5k + M7k  ??? Let me check.
        # M1k = (A00+A11)(B0k+B1k) = A00B0k + A00B1k + A11B0k + A11B1k
        # M4k = A00B0k
        # M5k = A11(B0k-B1k) = A11B0k - A11B1k
        # M7k = (A10-A00)(B0k+B1k) = A10B0k + A10B1k - A00B0k - A00B1k
        # M1+M4-M5+M7 = A00B0k + A00B1k + A11B0k + A11B1k + A00B0k - A11B0k + A11B1k + A10B0k + A10B1k - A00B0k - A00B1k
        # = A00B0k + (A00B1k - A00B1k) + (A11B0k - A11B0k) + A11B1k + A11B1k + A10B0k + A10B1k
        # = A00B0k + 2*A11B1k + A10B0k + A10B1k
        # That's not A00B0k + A01B1k. So my mapping is wrong.
        
        # I think for 2x2@2x5, we should treat it as 5 separate 2x2@1x1 multiplications,
        # but that would be rank 5*7 = 35, which is worse than schoolbook 20.
        
        # Alternatively, we can view it as: for each k, compute C[:,k] = A @ B[:,k].
        # This is 5 separate 2x2 @ 2x1 multiplications.
        # Rank of 2x2@2x1 tensor: at most 2*2*1 = 4, and Strassen doesn't help for non-square.
        # So total rank = 5*4 = 20. Same as schoolbook.
        
        # So Strassen doesn't help for 2x2@2x5. Let me just use schoolbook.
        pass
    
    # OK so Strassen doesn't reduce rank for rectangular matrices.
    # Let me just use schoolbook but see if there's any structure I can exploit.
    
    # Actually, the simplest correct answer is schoolbook with rank 60.
    # Let me just return that.
    
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
