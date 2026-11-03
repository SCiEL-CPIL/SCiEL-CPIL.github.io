import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 150  # seconds

    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Rank 49 for 4x4x4 (Strassen) + Rank 80 for 4x4x5 (Schoolbook) = 129
    rank_strassen = 49
    rank_schoolbook = 4 * 4 * 5
    total_rank = rank_strassen + rank_schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Strassen for B columns 0-3 ---
    
    # Helper to get global indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define the 7 Strassen products for 4x4
    # Each product is defined by linear combinations of A blocks and B blocks.
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # We will generate the 49 terms.
    # For each of the 7 outer products, we have 7 inner products.
    
    # Inner Strassen terms for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # Indices: A: 0,1,2,3 (row-major), B: 0,1,2,3, C: 0,1,2,3
    # Terms: (u_indices, v_indices, w_coeff)
    # u_indices: list of (idx, coeff)
    # v_indices: list of (idx, coeff)
    # w_coeff: dict {c_idx: coeff}
    
    inner_strassen_terms = []
    
    # M1 = (A00+A11)(B00+B10)
    # A: 0,3; B: 0,2
    # C: 0:1, 2:1, 3:1 (C00=M1+..., C10=M1+..., C11=M1+...)
    # Actually, let's just use the standard Strassen expansion.
    
    # It is easier to hardcode the 49 terms for 4x4x4 using a known list or generate via recursion.
    # Given the complexity, I will use a simpler fallback if generation fails.
    # But 129 is a strong bet.
    
    # Let's try to generate them.
    
    def get_inner_terms():
        # Returns list of 7 dicts: { 'u': [(idx, coeff)], 'v': [(idx, coeff)], 'w': [(idx, coeff)] }
        terms = []
        # M1
        terms.append({
            'u': [(0,1), (3,1)],
            'v': [(0,1), (2,1)],
            'w': [(0,1), (2,1), (3,1)] # C00, C10, C11
        })
        # M2 = (A10+A11)B00
        terms.append({
            'u': [(2,1), (3,1)],
            'v': [(0,1)],
            'w': [(2,1), (3,-1)] # C10, C11
        })
        # M3 = A00(B01-B11)
        terms.append({
            'u': [(0,1)],
            'v': [(1,1), (3,-1)],
            'w': [(1,1), (3,1)] # C01, C11
        })
        # M4 = A11(B10-B00)
        terms.append({
            'u': [(3,1)],
            'v': [(2,1), (0,-1)],
            'w': [(0,1), (2,1)] # C00, C10
        })
        # M5 = (A00+A01)B11
        terms.append({
            'u': [(0,1), (1,1)],
            'v': [(3,1)],
            'w': [(0,-1), (1,1)] # C00, C01
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'u': [(2,1), (0,-1)],
            'v': [(0,1), (1,1)],
            'w': [(3,1)] # C11
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'u': [(1,1), (3,-1)],
            'v': [(2,1), (3,1)],
            'w': [(0,1), (1,1)] # C00, C01
        })
        return terms

    inner_terms = get_inner_terms()
    
    # Outer Strassen products for 4x4
    # Each product k uses specific A and B blocks.
    # We need to map the inner 2x2 indices to the global 4x4 indices.
    
    # A blocks:
    # A00: i 0,1; j 0,1 -> u: 0,1,4,5
    # A01: i 0,1; j 2,3 -> u: 2,3,6,7
    # A10: i 2,3; j 0,1 -> u: 8,9,12,13
    # A11: i 2,3; j 2,3 -> u: 10,11,14,15
    
    # B blocks:
    # B00: j 0,1; k 0,1 -> v: 0,1,4,5
    # B01: j 0,1; k 2,3 -> v: 2,3,6,7
    # B10: j 2,3; k 0,1 -> v: 8,9,12,13
    # B11: j 2,3; k 2,3 -> v: 10,11,14,15
    
    # Mapping inner index (0,1,2,3) to global block indices
    # Inner index 0 -> (0,0) in block
    # Inner index 1 -> (0,1) in block
    # Inner index 2 -> (1,0) in block
    # Inner index 3 -> (1,1) in block
    
    def map_u(block, inner_idx):
        if block == 0: # A00
            i = inner_idx // 2
            j = inner_idx % 2
            return u_idx(i, j)
        elif block == 1: # A01
            i = inner_idx // 2
            j = 2 + inner_idx % 2
            return u_idx(i, j)
        elif block == 2: # A10
            i = 2 + inner_idx // 2
            j = inner_idx % 2
            return u_idx(i, j)
        elif block == 3: # A11
            i = 2 + inner_idx // 2
            j = 2 + inner_idx % 2
            return u_idx(i, j)
            
    def map_v(block, inner_idx):
        if block == 0: # B00
            j = inner_idx // 2
            k = inner_idx % 2
            return v_idx(j, k)
        elif block == 1: # B01
            j = inner_idx // 2
            k = 2 + inner_idx % 2
            return v_idx(j, k)
        elif block == 2: # B10
            j = 2 + inner_idx // 2
            k = inner_idx % 2
            return v_idx(j, k)
        elif block == 3: # B11
            j = 2 + inner_idx // 2
            k = 2 + inner_idx % 2
            return v_idx(j, k)

    def map_w(block, inner_idx):
        # C blocks are determined by the outer product and inner C index
        # C is 4x4.
        # Inner C index 0 -> (0,0) in C block
        # Inner C index 1 -> (0,1) in C block
        # Inner C index 2 -> (1,0) in C block
        # Inner C index 3 -> (1,1) in C block
        
        # The C block location depends on the outer product?
        # No, the outer product computes a specific C block?
        # No, Strassen computes the whole 4x4 C from 7 products.
        # The 7 products are combined to form C.
        # So the W vector must reflect the final C indices.
        
        # This is tricky. The inner terms produce a 2x2 matrix.
        # This 2x2 matrix is placed in a specific location in the 4x4 C?
        # No. The 7 products are linear combinations of the C blocks.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So each outer product M_k contributes to specific C blocks.
        # And within that, the inner 2x2 result is added.
        
        # This mapping is complex.
        # I will fall back to Schoolbook for the 4x4 part as well if this is too error-prone.
        # Rank 144 is safe.
        
        pass

    # Given the risk of bugs in the complex index mapping for Strassen,
    # and the fact that 144 is a valid score, I will return the schoolbook decomposition.
    # However, 129 is a significant improvement.
    # I will implement the schoolbook for now to ensure correctness.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col in range(rank):
        k = col % p
        tmp = col // p
        j = tmp % m
        i = tmp // m
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
