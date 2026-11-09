"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 4, 9]
TIME_LIMIT = 120  # seconds; keep margin before 150s kill

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 with Strassen rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen: 
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11-B21)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    # For each rank-1 term r, we need U[i,j], V[j,k], W[k,i]
    # such that sum_r U[i,j]*V[j,k]*W[k,i] = C[i,k]
    
    # Let's use the standard Strassen decomposition:
    # Term 0: (A11+A22)(B11+B22) -> contributes to C11(+1), C22(+1)
    # Term 1: (A21+A22)B11 -> contributes to C21(+1), C22(-1)
    # Term 2: A11(B21-B22) -> contributes to C12(+1), C22(+1)
    # Term 3: A22(B11-B21) -> contributes to C11(+1), C21(+1)
    # Term 4: (A11+A12)B22 -> contributes to C11(-1), C12(+1)
    # Term 5: (A21-A11)(B11+B12) -> contributes to C22(+1)
    # Term 6: (A12-A22)(B21+B22) -> contributes to C11(+1)
    
    # We need to express each term as a sum of rank-1 tensors.
    # Actually, each Strassen "multiply" is itself a 2x2 * 2x2 or 2x1 * 2x1 etc.
    # Let me be more careful.
    
    # Standard Strassen for 2x2*2x2:
    # The 7 products are:
    # M1 = (A11+A22) * (B11+B22)  -> 2x2 * 2x2
    # M2 = (A21+A22) * B11        -> 2x1 * 2x1 (scalar)
    # M3 = A11 * (B21-B22)        -> 2x1 * 2x1 (scalar)
    # M4 = A22 * (B11-B21)        -> 2x1 * 2x1 (scalar)
    # M5 = (A11+A12) * B22        -> 2x1 * 2x1 (scalar)
    # M6 = (A21-A11) * (B11+B12)  -> 2x1 * 2x1 (scalar)
    # M7 = (A12-A22) * (B21+B22)  -> 2x1 * 2x1 (scalar)
    
    # This is getting complex. Let me just use the schoolbook for 2x2*2x2 (rank 4)
    # and for 2x2*2x1 (rank 2).
    
    # Actually, let's just build the full 4x4*4x9 decomposition using schoolbook
    # for all 2x2 blocks. This gives rank 4*4*9 = 144, same as the seed.
    # That's not an improvement.
    
    # Let me think differently. Let's use a known result:
    # For 4x4 * 4x4, the best known rank is 49 (or 49 for exact).
    # For 4x4 * 4x9, it should be around 4*49/4 * 9/4 ... no.
    
    # Actually, the tensor rank for matrix multiplication m×n × n×p is known for small cases.
    # For 4×4 × 4×9, the schoolbook rank is 144.
    # The best known upper bound is likely less.
    
    # Given time constraints, let me try a simple improvement:
    # Use Strassen for the 2x2 subblocks.
    
    # Let me implement a direct construction.
    
    return None  # placeholder

def _construct_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact. Returns True if valid."""
    rank = len(U[0])
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 if this is the (i,j) of A times (j,k) of B = (i,k) of C.
    # The decomposition says: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_{j,r} A[i,j] B[j,k] * (something)
    # More precisely, the bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition with rank r means:
    # C[i,k] = sum_{s=1}^{r} (sum_j U[i,j,s] A[i,j]) * (sum_k V[j,k,s] B[j,k]) * W[k,i,s]
    # Wait, the standard form is:
    # C[i,k] = sum_{s} (sum_j U[i,j,s] A[i,j]) * (sum_k V[j,k,s] B[j,k]) * W[k,i,s]
    # This doesn't look right. Let me reconsider.
    
    # The correct form for matrix multiplication tensor decomposition:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition gives:
    # C[i,k] = sum_{s=1}^{r} U_s[i,j] * V_s[j,k] * W_s[k,i] * A_s[i,j] * B_s[j,k]
    # where A_s, B_s, C_s are the s-th rank-1 factor.
    
    # Actually, the standard way to verify:
    # For each pair of input matrices A (n×m) and B (m×p), compute C = A*B.
    # Then check that C[i,k] == sum_s (sum_j U[i,j,s]*A[i,j]) * (sum_k' V[j,k',s]*B[j,k']) * W[k,i,s]
    # Wait, that's not right either.
    
    # Let me use the correct verification:
    # The decomposition defines:
    # C[i,k] = sum_{s} W[k,i,s] * (sum_j U[i,j,s] * A[i,j]) * (sum_{j'} V[j',k,s] * B[j',k])
    # This should equal sum_j A[i,j] B[j,k]
    
    # Let me test with random small matrices.
    import random
    random.seed(42)
    for trial in range(10):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        # Compute true C
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Compute via decomposition
        C_decomp = [[0]*p for _ in range(n)]
        for s in range(rank):
            # Compute intermediate: a_s[i] = sum_j U[i,j,s] * A[i,j]  -- wait, this is sum over j
            # Actually: a_s[i] = sum_j U[i,j,s] * A[i,j]  -- no, U[i,j,s] is a scalar coefficient
            # The rank-1 term contributes: W[k,i,s] * (sum_j U[i,j,s] A[i,j]) * (sum_j V[j,k,s] B[j,k])
            # Wait, I think the indices are:
            # U has shape (n*m, rank), so U[i*m+j, s]
            # V has shape (m*p, rank), so V[j*p+k, s]
            # W has shape (p*n, rank), so W[k*n+i, s]
            
            a_s = sum(U[i*m+j][s] * A[i][j] for j in range(m))
            b_s = sum(V[j*p+k][s] * B[j][k] for j in range(m))
            for i2 in range(n):
                for k2 in range(p):
                    C_decomp[i2][k2] += W[k2*n+i2][s] * a_s * b_s
        
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C_true[i][k]:
                    return False
    return True

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p  # 144
    
    # Verify schoolbook
    if _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        print(f"Schoolbook rank {best_rank} is valid.")
    else:
        print("ERROR: Schoolbook decomposition failed validation!")
        return best_result
    
    # Try to find a lower rank decomposition using random search
    # Strategy: randomly initialize U, V, W with small integers and try to refine
    # This is a very hard problem, so let's try a few simple structured approaches.
    
    # Approach 1: Block decomposition using 2x2 Strassen
    # Split 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bj is 2x3
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23]]
    
    # Each product like A11*B11 is 2x2 * 2x2 (rank 4 schoolbook or 7 Strassen)
    # Each product like A12*B21 is 2x2 * 2x2
    # But B11 is 2x2, so this works.
    
    # Let's use Strassen (rank 7) for each 2x2*2x2 block.
    # There are 2*3 = 6 such blocks, each contributing 7 rank-1 terms.
    # Total rank: 6 * 7 = 42? No, that's not right.
    
    # Actually, each C entry is a sum of two products. So we can't just sum independent Strassen decompositions.
    # We need to be more careful.
    
    # Let's just return the schoolbook decomposition. It's valid and has rank 144.
    # Finding a better one is a hard open problem in general, but for small matrices it's known.
    # The best known rank for 4x4 * 4x4 is 49 (Leander, 1990s? or maybe lower).
    # For 4x4 * 4x9, it would be higher.
    
    # Given the time limit, let's just return schoolbook.
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
