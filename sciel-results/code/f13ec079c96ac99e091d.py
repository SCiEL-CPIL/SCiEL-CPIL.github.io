import time

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    # Part 1: j in {0, 1}
    # k pairs: (0,1), (2,3), (4,5), (6,7)
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for k0, k1 in k_pairs:
        # Strassen for 2x2 * 2x2
        # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B indices: (0,k0)->k0, (0,k1)->k1, (1,k0)->9+k0, (1,k1)->9+k1
        # W indices: (k0,0)->2k0, (k0,1)->2k0+1, (k1,0)->2k1, (k1,1)->2k1+1
        
        # P1 = a00 * (b1k0 + b1k1)
        # Term 1: a00 * b1k0
        add_term(0, 1, 9+k0, 1, 2*k0, 1)
        # Term 2: a00 * b1k1
        add_term(0, 1, 9+k1, 1, 2*k1, 1)
        # Wait, Strassen P1 is a00 * (b10 + b11).
        # In tensor decomposition, we sum rank-1 tensors.
        # P1 contributes to C00, C11.
        # C00 += P1, C11 += P1.
        # So we need to add P1 to W[2k0] (C0k0) and W[2k1+1] (C1k1)?
        # No, the output indices are fixed.
        # Let's trace the contributions carefully.
        
        # C_0k0 = P1 + P4 - P6 + P7
        # C_0k1 = P5 + P6
        # C_1k0 = P2 + P3
        # C_1k1 = P1 - P3 + P4 + P5
        
        # We define 7 intermediate products P1..P7.
        # Each Pr is a product of a linear combo of A and a linear combo of B.
        # In the decomposition, each term is u_alpha * v_alpha * w_alpha.
        # We can implement the Strassen terms directly.
        
        # P1 = a00 * (b1k0 + b1k1)
        # This is not a single rank-1 term in the original basis, it's a sum.
        # However, we can decompose P1 into two terms:
        # T1: a00 * b1k0
        # T2: a00 * b1k1
        # But Strassen combines them.
        # Actually, the rank-1 terms in the decomposition correspond to the "multiplications" in the algorithm.
        # If we use Strassen, we have 7 multiplications.
        # Multiplication 1: M1 = a00 * (b1k0 + b1k1).
        # This M1 contributes to C0k0 and C1k1.
        # So we need a term where U has a00, V has (b1k0+b1k1), W has (1 for C0k0, 1 for C1k1).
        # But V must be a vector with entries for each B element.
        # So for M1:
        # U[0] += 1 (a00)
        # V[9+k0] += 1, V[9+k1] += 1
        # W[2*k0] += 1 (C0k0), W[2*k1+1] += 1 (C1k1)
        
        # Let's implement this way.
        
        # M1 = a00 * (b1k0 + b1k1)
        add_term(0, 1, 9+k0, 1, 2*k0, 1)
        U[0][c-1] = 1 # Already set
        V[9+k1][c-1] = 1 # Add second part of V
        W[2*k1+1][c-1] = 1 # Add second part of W
        # Wait, add_term increments c. I need to modify the last column added.
        # Better to just call add_term for each component? No, that increases rank.
        # I need to accumulate into the same column index.
        
        # Let's refactor add_term to not increment, or manage index manually.
        
        pass

    # Refactored approach:
    # We will manually fill the columns.
    
    c = 0
    def add_to_col(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val

    def new_col():
        nonlocal c
        c += 1

    # Part 1: j in {0, 1}
    for k0, k1 in k_pairs:
        # M1 = a00 * (b1k0 + b1k1)
        # Contributes to C0k0 (+1) and C1k1 (+1)
        add_to_col(0, 1, 9+k0, 1, 2*k0, 1)
        add_to_col(0, 0, 9+k1, 1, 2*k1+1, 1)
        new_col()
        
        # M2 = (a00 + a01) * b0k0
        # Contributes to C1k0 (+1)
        add_to_col(0, 1, 9*0+k0, 1, 2*k0+1, 1)
        add_to_col(1, 1, 9*0+k0, 0, 2*k0+1, 0) # U[1] is a01, but V is same, so we add to same col?
        # Wait, M2 = (a00+a01) * b0k0.
        # U[0]=1, U[1]=1. V[9*0+k0]=1. W[2*k0+1]=1.
        # My add_to_col adds to the current column c.
        # So:
        # U[0][c] += 1
        # U[1][c] += 1
        # V[k0][c] += 1
        # W[2*k0+1][c] += 1
        
        # Let's restart the filling logic to be cleaner.
        
        pass

    # Correct Implementation
    
    c = 0
    def start_term():
        nonlocal c
        return c

    def end_term():
        nonlocal c
        c += 1

    def set_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    for k0, k1 in k_pairs:
        # M1 = a00 * (b1k0 + b1k1) -> C0k0, C1k1
        col = start_term()
        set_term(0, 1, 9+k0, 1, 2*k0, 1, col)
        set_term(0, 0, 9+k1, 1, 2*k1+1, 1, col)
        end_term()
        
        # M2 = (a00 + a01) * b0k0 -> C1k0
        col = start_term()
        set_term(0, 1, k0, 1, 2*k0+1, 1, col)
        set_term(1, 1, k0, 0, 2*k0+1, 0, col)
        end_term()
        
        # M3 = (a10 + a11) * b1k1 -> C1k0
        col = start_term()
        set_term(2, 1, 9+k1, 1, 2*k0+1, 1, col)
        set_term(3, 1, 9+k1, 0, 2*k0+1, 0, col)
        end_term()
        
        # M4 = a11 * (b0k0 - b1k0) -> C0k0, C1k1
        col = start_term()
        set_term(3, 1, k0, 1, 2*k0, 1, col)
        set_term(3, 0, 9+k0, -1, 2*k1+1, 1, col)
        end_term()
        
        # M5 = (a00 + a10) * b0k1 -> C0k1, C1k1
        col = start_term()
        set_term(0, 1, k1, 1, 2*k1, 1, col)
        set_term(2, 1, k1, 0, 2*k1+1, 1, col)
        end_term()
        
        # M6 = (a01 - a10) * (b1k0 + b1k1) -> C0k1, C0k0
        # C0k1 += M6, C0k0 -= M6
        col = start_term()
        set_term(1, 1, 9+k0, 1, 2*k1, 1, col)
        set_term(2, -1, 9+k0, 0, 2*k0, -1, col)
        set_term(1, 0, 9+k1, 1, 2*k1, 0, col) # Wait, V is (b1k0+b1k1)
        set_term(2, 0, 9+k1, 1, 2*k0, 0, col)
        end_term()
        
        # M7 = (a00 - a01) * (b0k0 + b0k1) -> C0k0, C1k1
        # C0k0 += M7, C1k1 += M7
        col = start_term()
        set_term(0, 1, k0, 1, 2*k0, 1, col)
        set_term(1, -1, k0, 0, 2*k1+1, 1, col)
        set_term(0, 0, k1, 1, 2*k0, 0, col)
        set_term(1, 0, k1, 1, 2*k1+1, 0, col)
        end_term()

    # k=8 (2x2x1)
    k8 = 8
    # C08 = a00*b08 + a01*b18
    # C18 = a10*b08 + a11*b18
    
    col = start_term()
    set_term(0, 1, k8, 1, 2*k8, 1, col)
    end_term()
    
    col = start_term()
    set_term(1, 1, 9+k8, 1, 2*k8, 1, col)
    end_term()
    
    col = start_term()
    set_term(2, 1, k8, 1, 2*k8+1, 1, col)
    end_term()
    
    col = start_term()
    set_term(3, 1, 9+k8, 1, 2*k8+1, 1, col)
    end_term()
    
    # Part 2: j=2
    for i in range(2):
        for k in range(9):
            col = start_term()
            set_term(i*3+2, 1, 2*9+k, 1, 2*k+i, 1, col)
            end_term()
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    try:
        U, V, W = _make_2x3x9_rank_50()
        if time.time() > deadline:
            raise TimeoutError()
        return {"U": U, "V": V, "W": W}
    except Exception:
        n, m, p = 2, 3, 9
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
# EVOLVE-BLOCK-END
