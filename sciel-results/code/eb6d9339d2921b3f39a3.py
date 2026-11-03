import time
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # Leave 20s safety margin

    n, m, p = DIMENSIONS
    # U: (n*m) x rank
    # V: (m*p) x rank  
    # W: (p*n) x rank
    
    # Use rank-84 decomposition
    # Strategy: 4x4x4 (rank 49) + 4x4x2 (rank 35)
    
    # First, let's use a known rank-49 decomposition for 4x4x4
    # Then handle the remaining 2 columns
    
    # For simplicity, use a structured rank-84 decomposition
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # For the first 4 columns of the result (4x4x4 part), use rank-49
    # For the last 2 columns (4x4x2 part), use rank-35
    
    # Use a simple decomposition: for each of the 4x4x4, use Strassen-like
    # Actually, let's use a more straightforward approach
    
    # Decomposition for 4x4x4 with rank 49 (known optimal)
    # We'll use the decomposition from the literature
    
    # For now, use a practical rank-84 decomposition
    # Each term: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's construct a valid rank-84 decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                # Only use 84 terms instead of 96
                # Skip 12 terms strategically
                key = (i, j, k)
                # Skip when i==3 and j==3 and k>=4 (last 2)
                # and a few more to get to 84
                if i == 3 and j == 3 and k >= 4:
                    continue
                if i == 2 and j == 3 and k == 5:
                    continue
                if i == 1 and j == 3 and k == 5:
                    continue
                if i == 0 and j == 3 and k == 5:
                    continue
                if i == 3 and j == 2 and k == 5:
                    continue
                if i == 3 and j == 1 and k == 5:
                    continue
                if i == 3 and j == 0 and k == 5:
                    continue
                if i == 2 and j == 2 and k == 5:
                    continue
                if i == 2 and j == 1 and k == 5:
                    continue
                if i == 2 and j == 0 and k == 5:
                    continue
                if i == 1 and j == 1 and k == 5:
                    continue
                if i == 1 and j == 0 and k == 5:
                    continue
                if i == 0 and j == 0 and k == 5:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify we have exactly 84 columns
    # If not, we need a different approach
    # Let's use a proper rank-84 decomposition
    
    # Actually, the above won't give a valid decomposition since we're skipping terms
    # We need a proper algebraic decomposition
    
    # Let's use a different strategy: rank-80 decomposition
    # Using 4x4x4 with rank 49 and 4x4x2 with rank 31
    
    # For a correct approach, let's use the following:
    # The 4x4x4 multiplication can be done in rank 49
    # The 4x4x2 multiplication (4x4 times 4x2 = 4x2) can be done in rank 32
    
    # Total: 49 + 32 = 81
    
    # But we need to verify the decomposition is correct
    
    # Given the complexity, let's return a known valid rank-84 decomposition
    # that's been verified to work
    
    # Use the decomposition where we combine terms
    # For a 4x4x6 multiplication, one valid rank-84 decomposition is:
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Terms 0-48: 4x4x4 part (rank 49)
    # Use Strassen-based decomposition for 4x4x4
    # A = [[a11, a12, a13, a14],
    #      [a21, a22, a23, a24],
    #      [a31, a32, a33, a34],
    #      [a41, a42, a43, a44]]
    # B = [[b11, b12, b13, b14],
    #      [b21, b22, b23, b24],
    #      [b31, b32, b33, b34],
    #      [b41, b42, b43, b44]]
    
    # For simplicity, use schoolbook for the 4x4x4 part (rank 64) 
    # and schoolbook for 4x4x2 (rank 32), total 96 - not helpful
    
    # Let's use a verified rank-84 decomposition
    # The following is a known rank-84 decomposition for 4x4x6
    
    # Actually, let's just use rank 84 with a valid construction
    # We'll use the fact that we can merge some terms
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                if col < rank:
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
    
    # This gives us 96 terms, but we only filled 84
    # This is WRONG - we need exactly 84 valid terms
    
    # Let me restart with a proper approach
    # Use rank 96 (schoolbook) as fallback, but try to find better
    
    # Given time constraints, return a valid decomposition
    # The safest is to use the schoolbook rank-96
    rank = 96
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
# EVOLVE-BLOCK-END
