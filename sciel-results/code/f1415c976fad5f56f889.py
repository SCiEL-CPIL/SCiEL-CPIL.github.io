"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    # Indexing:
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[k,i] (trace convention: C[i,k] is at index k*n+i in W)
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means W is indexed such that the output element C[i,k] is associated with index k*n + i.
    # Let's verify the tensor equation:
    # T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
    # In matrix multiplication C = A B:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So we need: sum_j A_{i,j} B_{j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r ?
    # This looks like a permutation.
    # Standard tensor decomposition for matrix multiplication M_{n,m,p}:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k}.
    # Decomposition: T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
    # Here U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # U index: i*m + j.
    # V index: j*p + k.
    # W index: k*n + i.
    
    # Schoolbook rank: n*m*p = 4*4*6 = 96.
    # We want to find a lower rank.
    # Known ranks:
    # 2x2x2: 7 (Strassen)
    # 3x3x3: 23 (Winograd) or 19 (Lader)
    # 4x4x4: 21
    # 4x4x6: ?
    
    # Strategy:
    # 1. Try to construct a valid decomposition with rank < 96.
    # 2. Use a known decomposition for 4x4x4 (rank 21) and 4x4x2 (rank ?).
    #    If 4x4x2 can be done in rank 16, total is 37.
    #    If 4x4x2 is done in rank 32 (schoolbook), total is 53.
    #    Let's check if 4x4x2 rank is 16.
    #    Lower bound for 4x4x2 is max(4*4, 4*2, 4*2) = 16? No, slice rank bound is max(16, 8, 8) = 16.
    #    Is rank 16 achievable?
    #    Yes, by treating it as two 4x4x1 multiplications? No, that's not independent.
    #    Actually, 4x4x2 is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
    #    This can be seen as two 4x4x1 multiplications sharing A.
    #    However, the tensor rank is not simply additive if they share indices.
    #    But if we view the 4x4x2 tensor as having disjoint supports in the last index (k=0,1),
    #    we can decompose it as the sum of two 4x4x1 tensors.
    #    rank(4x4x1) = 16 (since C = A * b, this is a linear map from A to C, which is 16-dimensional).
    #    Actually, rank(4x4x1) is 16.
    #    So rank(4x4x2) <= 16 + 16 = 32.
    #    Can it be less?
    #    The lower bound is 16.
    #    It is known that rank(4x4x2) = 16?
    #    Let's assume we can get 4x4x4 in 21 and 4x4x2 in 16. Total 37.
    #    If 4x4x2 is 32, total 53.
    
    # Let's try to find a decomposition using a numerical solver or random search with verification.
    # Given the time limit, a random search for rank 37-50 might be slow.
    # A safer bet is to use the Strassen-based decomposition for 4x4x4 (rank 21 is hard to hardcode without error).
    # Let's use the rank-21 decomposition for 4x4x4 if I can find a reliable source or derive it.
    # Alternatively, use the rank-56 Strassen decomposition for 4x4x4 and rank-32 for 4x4x2. Total 88.
    # 88 < 96, so it's an improvement.
    
    # Let's implement the rank-88 decomposition (56 + 32).
    # This is robust and likely valid.
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook fallback
    U_school = np.zeros((16, 96), dtype=int)
    V_school = np.zeros((24, 96), dtype=int)
    W_school = np.zeros((24, 96), dtype=int)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U_school[i*4+j, col] = 1
                V_school[j*6+k, col] = 1
                W_school[k*4+i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = 96
    
    # Try to find a better decomposition using structured Strassen for 4x4x4
    # and schoolbook for 4x4x2.
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M7
    
    # We need to map this to the 4x4x4 tensor.
    # Split A, B, C into 2x2 blocks.
    # A: 2x2 blocks of 2x2.
    # B: 2x2 blocks of 2x2.
    # C: 2x2 blocks of 2x2.
    
    # There are 4 2x2x2 multiplications needed for the 4x4x4?
    # No, each 2x2 block of C is a sum of two 2x2 products.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each product (e.g., A00*B00) is a 2x2x2 tensor.
    # We can use Strassen for each 2x2x2.
    # But we can share terms if we group them.
    # For C00, we have two products: P1 = A00*B00, P2 = A01*B10.
    # We can compute P1 and P2 separately using Strassen?
    # That would be 7 + 7 = 14 terms for C00.
    # Total for 4 blocks: 4 * 14 = 56 terms.
    # This matches the rank 56 estimate.
    
    # Let's implement this.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_strassen_term(a_i, a_j, b_i, b_j, c_i, c_j, u_val, v_val, w_val):
        # a_i, a_j: block indices for A (0 or 1 for row/col block)
        # b_i, b_j: block indices for B
        # c_i, c_j: block indices for C
        # u_val, v_val, w_val: values for this term in the Strassen decomposition
        
        # Map block indices to global indices
        # A global index: (a_i*2 + local_i, a_j*2 + local_j)
        # B global index: (b_i*2 + local_i, b_j*2 + local_j)
        # C global index: (c_i*2 + local_i, c_j*2 + local_j)
        
        # We need to add 7 terms for the Strassen decomposition of the 2x2x2 block.
        # The local indices are 0,1.
        
        # M1
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
        # M2
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        # M3
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * -1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+1), w_val * 1 ))
        
        # M4
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+0), w_val * -1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * -1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+0), w_val * 1 ))
        
        # M5
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+0)*4 + (c_j*2+1), w_val * 1 ))
        
        # M6
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+0), u_val * 1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * -1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+0), u_val * -1 ))
        V_terms.append(( (b_i*2+0)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+0), w_val * 1 ))
        
        # M7
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+0)*4 + (a_j*2+1), u_val * 1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * -1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+0), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
        U_terms.append(( (a_i*2+1)*4 + (a_j*2+1), u_val * -1 ))
        V_terms.append(( (b_i*2+1)*6 + (b_j*2+1), v_val * 1 ))
        W_terms.append(( (c_i*2+1)*4 + (c_j*2+1), w_val * 1 ))
        
    # This manual expansion is very error-prone.
    # Let's use a simpler approach: just use schoolbook for 4x4x4 and 4x4x2.
    # Rank 96.
    # Then try to optimize using a random search with a lower rank target.
    
    # Let's try a random search for rank 40-95.
    # We will use a verification function.
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    val = np.dot(U[i*4+j], np.multiply(V[j*6+k], W[k*4+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try random search
    np.random.seed(42)
    for target_rank in range(40, 96, 5):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            U = np.zeros((16, target_rank))
            V = np.zeros((24, target_rank))
            W = np.zeros((24, target_rank))
            
            for r in range(target_rank):
                # Random sparse initialization
                i = np.random.randint(0, 4)
                j = np.random.randint(0, 4)
                k = np.random.randint(0, 6)
                
                U[i*4+j, r] = np.random.choice([-1, 1])
                V[j*6+k, r] = np.random.choice([-1, 1])
                W[k*4+i, r] = np.random.choice([-1, 1])
                
                # Add some more terms
                for _ in range(3):
                    i2 = np.random.randint(0, 4)
                    j2 = np.random.randint(0, 4)
                    k2 = np.random.randint(0, 6)
                    U[i2*4+j2, r] += np.random.choice([-1, 0, 1])
                    V[j2*6+k2, r] += np.random.choice([-1, 0, 1])
                    W[k2*4+i2, r] += np.random.choice([-1, 0, 1])
            
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        if best_rank < 96:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
