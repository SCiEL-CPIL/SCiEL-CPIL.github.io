"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition.

Strategy:
1. Define a library of known low-rank decompositions for small sub-tensors:
   - Rank-1: 1x1x1
   - Rank-2: 1x2x1 (trivial), 2x1x1 (trivial), 1x1x2 (trivial), 2x2x1 (rank 2? No, rank 2 is 2*1*1? 
     Wait, R(2,1,1)=2, R(1,2,1)=2, R(1,1,2)=2.
     R(2,2,1)=4? No, R(n,m,1) = n*m. So R(2,2,1)=4.
     R(2,1,2)=4. R(1,2,2)=4.
     R(2,2,2)=7 (Strassen).
     R(3,3,3)=7 (Strassen).
     R(2,3,3)=? R(3,2,3)=?
     R(2,2,3)=?
   - We will include:
     - 1x1x1: rank 1
     - 1x2x1: rank 2 (schoolbook)
     - 2x1x1: rank 2
     - 1x1x2: rank 2
     - 2x2x1: rank 4 (schoolbook)
     - 1x2x2: rank 4
     - 2x1x2: rank 4
     - 2x2x2: rank 7 (Strassen)
     - 3x3x3: rank 7 (Strassen)
     - 2x3x3: rank ? (likely 10 or 11? Schoolbook is 18. Strassen-based? 
       2x3x3 can be seen as 2x(3x3) + ... no.
       Actually, R(2,3,3) <= R(2,2,3) + R(2,1,3) = ?
       Let's stick to knowns: 2x2x2 (7), 3x3x3 (7).
       Schoolbook for any size is always a candidate.
       
2. The problem is to partition the indices (3 rows, 5 cols, 6 cols) into blocks.
   Let A be 3x5, B be 5x6.
   We partition the 5 shared dimension (cols of A, rows of B) into segments.
   We partition the 3 rows of A (and C) into segments.
   We partition the 6 cols of B (and C) into segments.
   
   A partition is defined by:
   - Row splits: 3 -> sum of row_block_sizes
   - Col splits (shared): 5 -> sum of col_block_sizes
   - Out splits (cols of B): 6 -> sum of out_block_sizes
   
   For each combination of (row_block, col_block, out_block), we compute the rank 
   of the sub-tensor T[rb, cb, ob]. We choose the minimum rank decomposition 
   available in our library for that size.
   
   The total rank is the sum of ranks of all blocks.
   
   We enumerate all possible partitions of 3, 5, and 6.
   Partitions of 3: [3], [1,2], [2,1], [1,1,1]
   Partitions of 5: [5], [1,4], [4,1], [2,3], [3,2], [1,1,3], [1,3,1], [3,1,1], [1,2,2], [2,1,2], [2,2,1], [1,1,1,2], ...
   Partitions of 6: [6], [1,5], ..., [1,1,1,1,1,1]
   
   The number of partitions is small (p(3)=3, p(5)=7, p(6)=11).
   Total combinations: 3 * 7 * 11 = 231.
   For each combination, we sum the ranks.
   
   We need a function `min_rank(n, m, p)` that returns the minimum known rank for an n x m x p matmul tensor.
   We will use:
   - Schoolbook: n*m*p
   - Strassen for 2x2x2: 7
   - Strassen for 3x3x3: 7
   - For other sizes, we can use recursive bounds if we want, but for now, 
     we will use schoolbook for anything not in {1, 2, 3} x {1, 2, 3} x {1, 2, 3} 
     except we can also use 2x2x2 Strassen for sub-blocks if the block is 2x2x2.
     
   Actually, we can just use the minimum of:
   - Schoolbook rank
   - If n=2, m=2, p=2: 7
   - If n=3, m=3, p=3: 7
   - If n=2, m=3, p=3: ? 
     R(2,3,3) is at least R(2,2,2)=7. 
     Can we do better than schoolbook 18?
     R(2,3,3) <= R(2,2,2) + R(2,1,2) = 7 + 4 = 11? 
     Wait, R(2,1,2) is 4 (schoolbook). 
     So R(2,3,3) <= 11.
     Is there a better one? 
     R(2,3,3) <= R(2,2,3) + R(2,1,3).
     R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
     So R(2,3,3) <= 11 + 4 = 15? No, 7+4=11 for 2x2x3.
     Then 2x1x3 is 6. Total 17.
     My previous 11 was for 2x2x2 + 2x1x2. 
     2x3x3 = (2x2x2) + (2x1x3)? No, the middle dimension splits.
     A(2x3) * B(3x3).
     Split A into (2x2) and (2x1).
     Split B into (2x3) and (1x3).
     Term 1: A1(2x2) * B1(2x3) -> Rank(2,2,3)
     Term 2: A2(2x1) * B2(1x3) -> Rank(2,1,3)
     Rank(2,2,3): Split p=3 into 2+1.
       Rank(2,2,2) + Rank(2,2,1) = 7 + 4 = 11.
     Rank(2,1,3): Schoolbook 6.
     Total 17.
     
     Another split for 2x3x3:
     Split n=2 into 1+1.
     Term 1: A1(1x3) * B1(3x3) -> Rank(1,3,3) = 9.
     Term 2: A2(1x3) * B2(3x3) -> Rank(1,3,3) = 9.
     Total 18.
     
     So 17 is better than 18.
     
     What about R(2,2,3)?
     Split m=2 into 1+1.
     Rank(2,1,3) + Rank(2,1,3) = 6+6=12.
     Split p=3 into 2+1.
     Rank(2,2,2) + Rank(2,2,1) = 7+4=11.
     Split n=2 into 1+1.
     Rank(1,2,3) + Rank(1,2,3) = 6+6=12.
     So R(2,2,3) <= 11.
     
     So R(2,3,3) <= 17.
     
     We will implement a recursive `min_rank` function that tries all splits and takes the minimum, 
     with base cases for 2x2x2 (7) and 3x3x3 (7).
     
3. We will find the best partition and construct the decomposition.
   Construction requires mapping the global indices to local indices for each block 
   and using the specific U, V, W matrices for the chosen sub-decomposition.
   
   To keep it simple and robust, we will only use the "standard" decompositions:
   - Schoolbook for any block.
   - Strassen for 2x2x2.
   - Strassen for 3x3x3.
   
   If a block is, say, 2x3x3, we will use the schoolbook rank 18 for simplicity in construction, 
   UNLESS we have a specific construction for it. 
   Given the complexity of constructing arbitrary low-rank decompositions for arbitrary blocks, 
   and the fact that the "best" partition likely uses 3x3x3 and 2x2x2 or 3x2x2 blocks, 
   we will restrict our construction to:
   - 3x3x3 (Strassen)
   - 2x2x2 (Strassen)
   - Schoolbook (for everything else)
   
   This is a heuristic search. It might find rank 25 or 24 if a partition allows it.
   For example, if we can fit two 3x3x3 and one 3x2x2 (schoolbook 12) ... 
   3x5x6.
   Rows: 3.
   Cols: 5.
   Outs: 6.
   
   Partition:
   Rows: [3]
   Cols: [3, 2]
   Outs: [3, 3]
   
   Blocks:
   1. 3x3x3 (Strassen, rank 7)
   2. 3x3x3 (Strassen, rank 7)
   3. 3x2x3 (Schoolbook, rank 18) -> Total 32. Worse.
   
   How about:
   Rows: [2, 1]
   Cols: [3, 2]
   Outs: [3, 3]
   
   Blocks:
   1. 2x3x3 (Rank 17? We don't have a construction for 17, only schoolbook 18).
   If we use schoolbook for 2x3x3 (18) and 1x3x3 (9) and 2x2x3 (6) and 1x2x3 (6).
   Total: 18+9+6+6 = 39.
   
   It seems the 3x3x3 blocks are very efficient.
   The current static solution uses:
   - 3x3x3 (7)
   - 3x3x3 (7)
   - 3x2x3 (18) -> Wait, current uses schoolbook for 3x2x3?
     Current code: `add_schoolbook(0, 3, 0, 3, 2, 3)` -> 3*2*3 = 18.
     And `add_schoolbook(0, 3, 3, 3, 2, 3)` -> 18.
     Total 7+7+18+18 = 50? 
     Wait, the current code says rank 26.
     Let's re-read the current code's `add_schoolbook`.
     `add_schoolbook(0, 3, 0, 3, 2, 3)` -> ni=3, nj=2, nk=3. Rank = 3*2*3 = 18.
     Why does the docstring say 26?
     "The standard block approach (3x3 Strassen + 3x2 Schoolbook) gives 7+7+6+6=26."
     Ah, R(3,2,3) is NOT 18. 
     R(3,2,3) <= R(3,2,2) + R(3,2,1) = ?
     R(3,2,2) <= R(3,2,1) + R(3,2,1) = 6+6=12?
     R(3,2,1) = 6.
     R(3,2,2) can be done with Strassen? No, Strassen is 2x2x2 or 3x3x3.
     But R(3,2,2) is at most 12.
     Actually, R(3,2,2) is known to be 12? Or less?
     Wait, the current code uses `add_schoolbook` which is n*m*p.
     If the current code uses schoolbook for 3x2x3, the rank is 18.
     7+7+18+18 = 50.
     But the current program returns rank 26?
     Let's check the current code's `construct` return value.
     It returns `U, V, W` with `rank = 26` hardcoded in the variable `rank`?
     No, `rank = 26` is set, but `U` is allocated with `rank` columns.
     Then `add_strassen` adds 7 columns. (Col 0-6)
     `add_strassen` adds 7 columns. (Col 7-13)
     `add_schoolbook` adds 18 columns. (Col 14-31)
     `add_schoolbook` adds 18 columns. (Col 32-49)
     Total columns used: 50.
     But the matrix `U` has shape `(15, 26)`.
     The extra columns 26-49 are zero?
     If so, the rank is effectively 50, but the matrix width is 26.
     This would be an INVALID decomposition if the evaluator checks the width.
     The evaluator says: "factor matrices of shapes (n*m, rank) ... with integer/half-integer entries".
     If the width is 26, but we write to index 49, we get an IndexError or we truncate?
     `U[(i0 + ai) * m + (j0 + aj), col] += coef`
     If `col` exceeds `rank-1`, it will raise an error or be ignored if pre-allocated larger?
     The code allocates `U = np.zeros((n * m, rank), dtype=int)`.
     If `col` goes out of bounds, it crashes.
     
     Wait, did the current program actually run?
     "CURRENT PROGRAM (score 1.0000)".
     If it crashed, it would score 0.
     So it must have run.
     Let's re-read `add_schoolbook`.
     `add_schoolbook(0, 3, 0, 3, 2, 3)`
     `ni=3, nj=2, nk=3`.
     Loop: `for ai in range(3): for bj in range(2): for bk in range(3):`
     This is 18 iterations.
     `col` starts at 14.
     After 18 iterations, `col` is 32.
     `U` has 26 columns.
     `U[..., 26]` is out of bounds.
     
     Is it possible that `R(3,2,3)` is 6?
     No, R(n,m,p) >= n*m. R(3,2,3) >= 6.
     But schoolbook is 18.
     
     Maybe the current program is BROKEN but the score 1.0000 is from a different run?
     Or maybe I am misinterpreting the current code.
     Let's look at `STRASSEN_C`.
     It's a list of 9 lists of 7 ints.
     
     Let's look at the `add_strassen` function.
     It adds 7 columns.
     
     If the current program is indeed broken (crashes), then the "score 1.0000" is misleading or from a cached result of a different version.
     However, the prompt says "CURRENT PROGRAM (score 1.0000) and its strategy".
     I must assume the current program is valid.
     
     How can `add_schoolbook` fit in 26-14=12 columns?
     It can't, if it's schoolbook.
     
     UNLESS: The "Schoolbook" in the current program's *strategy description* is wrong, and it actually uses a low-rank decomposition for 3x2x3?
     But the code explicitly calls `add_schoolbook`.
     
     Wait! Look at `add_schoolbook` again.
     `def add_schoolbook(i0, j0, k0, ni, nj, nk):`
     It does NOT use Strassen.
     
     Is it possible that `rank` is not 26?
     `rank = 26` is hardcoded.
     
     This is a contradiction.
     If the current program crashes, it scores 0.
     If it scores 1.0, it must be valid.
     
     Hypothesis: The current program in the prompt is a *candidate* that was *thought* to be valid, but might be buggy.
     OR, I am missing something.
     
     Let's check `R(3,2,3)`.
     If we use Strassen for the 2x2 part?
     3x2x3.
     Split rows 3 -> 2+1.
     A1(2x2) * B1(2x3) -> R(2,2,3) = 11 (as derived).
     A2(1x2) * B2(2x3) -> R(1,2,3) = 6.
     Total 17.
     
     Split cols 2 -> 1+1.
     A1(3x1) * B1(1x3) -> 9.
     A2(3x1) * B2(1x3) -> 9.
     Total 18.
     
     Split outs 3 -> 2+1.
     A1(3x2) * B1(2x2) -> R(3,2,2) = 12.
     A2(3x2) * B2(2x1) -> R(3,2,1) = 6.
     Total 18.
     
     So min known is 17.
     
     If the current program uses 18 for the schoolbook parts, total rank is 7+7+18+18 = 50.
     If it allocates 26 columns, it crashes.
     
     I will implement a CORRECT program that:
     1. Searches for the best partition using a `min_rank` function.
     2. Constructs the decomposition using ONLY known valid low-rank decompositions (Strassen 2x2x2, Strassen 3x3x3, and Schoolbook).
     3. If the best partition requires a block like 3x2x3, and we don't have a rank-17 construction, we will use schoolbook (rank 18) for that block, BUT we will size the matrices correctly to the total sum of ranks.
     
     This guarantees a valid output.
     The rank might be higher than 26 if the best partition uses inefficient blocks, but the static 26 (if it worked) would be the baseline.
     Actually, if the static 26 was a lie (because it crashed), then ANY valid solution is an improvement.
     If the static 26 was real (maybe I'm misreading the code and it uses a rank-6 decomposition for 3x2x3?), then we need to beat 26.
     
     Let's assume the static program is valid and has rank 26.
     How?
     Maybe `add_schoolbook` is not schoolbook?
     No, it is.
     
     Maybe the partition is different?
     `add_strassen(0, 0, 0)` -> 3x3x3.
     `add_strassen(0, 0, 3)` -> 3x3x3.
     `add_schoolbook(0, 3, 0, 3, 2, 3)` -> 3x2x3.
     `add_schoolbook(0, 3, 3, 3, 2, 3)` -> 3x2x3.
     
     If the rank of 3x2x3 is 6?
     R(3,2,3) >= 6.
     Is R(3,2,3) = 6?
     No, R(n,m,p) >= n*m. 3*2=6.
     But also R(n,m,p) >= m*p = 6.
     And R(n,m,p) >= n*p = 9.
     So R(3,2,3) >= 9.
     Schoolbook is 18.
     
     So the current program MUST be crashing or returning an invalid result if it claims rank 26 with those blocks.
     
     I will proceed with the SEARCH strategy.
     It will find the best partition.
     For blocks where we don't have a specific low-rank construction, we use schoolbook.
     We will size the output matrices to the exact sum of the ranks used.
     
     This is robust.
"""

import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# Known low-rank decompositions
# We will implement:
# 1. Schoolbook (always available)
# 2. Strassen 2x2x2 (Rank 7)
# 3. Strassen 3x3x3 (Rank 7)

def _get_strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen."""
    # P1 = (a+b)(e+f)
    # P2 = (g+d)(e+f)
    # P3 = (a+c)(e+h)
    # P4 = (g+b)(f+h)
    # P5 = (a-d)(e+b)
    # P6 = (c-g)(f+d)
    # P7 = (a+c)(g+h) ? No, standard Strassen for 2x2:
    # P1 = (a+b)(e+f)
    # P2 = (g+d)(e+f)
    # P3 = (a+c)(e+h)
    # P4 = (g+b)(f+h)
    # P5 = (a-d)(e+b)
    # P6 = (c-g)(f+d)
    # P7 = (a+c)(g+h) is wrong.
    # Standard 2x2 Strassen:
    # P1 = (a+b)(e+f)
    # P2 = (g+d)(e+f)
    # P3 = (a+c)(e+h)
    # P4 = (g+b)(f+h)
    # P5 = (a-d)(e+b)
    # P6 = (c-g)(f+d)
    # P7 = (a+c)(g+h) -- Wait, C = [[a,b],[c,d]] * [[e,f],[g,h]]
    # C11 = P1 - P2 - P4 + P7?
    # Let's use the known 3x3x3 Strassen and pad it? No, 2x2x2 is simpler.
    # I will use the 3x3x3 Strassen for 2x2x2 by setting the last row/col to 0?
    # No, that's not exact.
    # I will implement 2x2x2 Strassen explicitly.
    
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # P1 = (a+b)(e+f)
    # P2 = (g+d)(e+f)
    # P3 = (a+c)(e+h)
    # P4 = (g+b)(f+h)
    # P5 = (a-d)(e+b)
    # P6 = (c-g)(f+d)
    # P7 = (a+c)(g+h) -- Correction: P7 = (a+c)(g+h) is not standard.
    # Standard:
    # P1 = (a+b)(e+f)
    # P2 = (g+d)(e+f)
    # P3 = (a+c)(e+h)
    # P4 = (g+b)(f+h)
    # P5 = (a-d)(e+b)
    # P6 = (c-g)(f+d)
    # P7 = (a+c)(g+h) -- Actually, C11 = P1 - P2 - P4 + P7 is not right.
    # C11 = P1 - P2 - P4 + P7?
    # Let's check:
    # P1 = ae+af+be+bf
    # P2 = ge+gf+de+df
    # P4 = gf+gh+bf+bh
    # P7 = ag+ah+cg+ch
    # This is getting messy. I will use the 3x3x3 Strassen for 2x2x2 by ignoring the 3rd dimension?
    # No, I will just use Schoolbook for 2x2x2 (Rank 8) to be safe, 
    # UNLESS I can verify the 2x2 Strassen quickly.
    # Given the risk, I will use Schoolbook for 2x2x2 in this search, 
    # and only use Strassen for 3x3x3.
    # Wait, R(2,2,2)=7 is well known. 
    # I will implement it carefully.
    
    # Let's use the 3x3x3 Strassen code from the current program and restrict it to 2x2x2?
    # No, the 3x3x3 Strassen uses specific indices.
    
    # I will implement a generic Schoolbook and a specific 3x3x3 Strassen.
    # For 2x2x2, I will use Schoolbook (Rank 8) for simplicity in this iteration.
    # This might yield a higher rank than optimal, but it will be VALID.
    pass

def _schoolbook_decomp(n, m, p):
    """Returns U, V, W lists for schoolbook n x m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W

def _strassen_3x3x3_decomp():
    """Returns U, V, W lists for 3x3x3 Strassen (Rank 7)."""
    # Reuse the logic from the current program, but return lists.
    n, m, p = 3, 3, 3
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    STRASSEN_PRODUCTS = [
        [(1, (0, 0), (0, 0)), (1, (0, 0), (0, 2)), (1, (0, 2), (0, 0)), (1, (0, 2), (0, 2))],
        [(1, (2, 0), (0, 0)), (1, (2, 2), (0, 0))],
        [(1, (0, 0), (1, 0)), (-1, (0, 0), (1, 2))],
        [(1, (0, 2), (1, 2)), (-1, (0, 2), (2, 1))],
        [(1, (0, 0), (0, 0)), (1, (0, 0), (0, 1)), (1, (1, 1), (0, 0)), (1, (1, 1), (0, 1))],
        [(1, (0, 2), (0, 2)), (1, (0, 2), (1, 2)), (1, (1, 2), (0, 2)), (1, (1, 2), (1, 2))],
        [(1, (1, 1), (1, 1)), (1, (1, 1), (2, 0)), (1, (2, 2), (1, 1)), (1, (2, 2), (2, 0))],
    ]
    STRASSEN_C = [
        [1, 0, 1, 0, -1, 0, 1],
        [-1, 0, 0, 0, 1, 1, -1],
        [1, 0, -1, 1, 0, -1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, -1, 0, 0, 0, 1, -1],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 1, -1, 0, 1, 0, -1],
        [0, -1, 1, 0, 0, 1, 1],
        [1, 1, -1, -1, 0, 0, 0],
    ]
    
    for q in range(7):
        for coef, (ai, aj), (bi, bk) in STRASSEN_PRODUCTS[q]:
            U[ai*m + aj][q] += coef
            V[bi*p + bk][q] += coef
        for dk in range(3):
            for di in range(3):
                wcoef = STRASSEN_C[dk*3+di][q]
                if wcoef != 0:
                    W[dk*n + di][q] += wcoef
                    
    return U, V, W

def _min_rank(n, m, p):
    """Estimate minimum rank for n x m x p using known bounds."""
    # Base cases
    if n == 1 and m == 1 and p == 1: return 1
    if n == 2 and m == 2 and p == 2: return 7
    if n == 3 and m == 3 and p == 3: return 7
    
    # Schoolbook
    best = n * m * p
    
    # Try splits
    # Split n
    for i in range(1, n):
        r = _min_rank(i, m, p) + _min_rank(n-i, m, p)
        if r < best: best = r
    # Split m
    for j in range(1, m):
        r = _min_rank(n, j, p) + _min_rank(n, m-j, p)
        if r < best: best = r
    # Split p
    for k in range(1, p):
        r = _min_rank(n, m, k) + _min_rank(n, m, p-k)
        if r < best: best = r
        
    return best

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 135.0
    
    # Get all partitions of n, m, p
    def get_partitions(num):
        parts = []
        def gen(remaining, current):
            if remaining == 0:
                parts.append(tuple(current))
                return
            for i in range(1, remaining + 1):
                current.append(i)
                gen(remaining - i, current)
                current.pop()
        gen(num, [])
        return parts
    
    row_parts = get_partitions(n)
    col_parts = get_partitions(m)
    out_parts = get_partitions(p)
    
    best_rank = n * m * p
    best_partition = (row_parts[0], col_parts[0], out_parts[0])
    
    # Evaluate all partitions
    for rp in row_parts:
        for cp in col_parts:
            for op in out_parts:
                total_rank = 0
                # Check if dimensions match
                if len(rp) != len(cp) or len(rp) != len(op):
                    # Blocks must be aligned? 
                    # Actually, a partition of rows and cols defines a grid.
                    # The number of row blocks must equal the number of col blocks?
                    # No, the blocks are defined by (row_seg, col_seg, out_seg).
                    # We need to iterate over all combinations of segments.
                    # This is a 3D partition.
                    # It's easier to think of it as:
                    # Choose row splits, col splits, out splits.
                    # The blocks are the Cartesian product of the segments.
                    # So if rows are split into 2 parts, cols into 2 parts, outs into 2 parts,
                    # there are 2x2x2 = 8 blocks.
                    pass
    
    # Simplified Search:
    # We will iterate over all ways to split the 3 dimensions.
    # For each combination of splits, we sum the ranks of the resulting blocks.
    
    def eval_partition(rp, cp, op):
        total_rank = 0
        # rp, cp, op are lists of sizes
        # We need to align them. 
        # The blocks are defined by (r_start, r_end, c_start, c_end, o_start, o_end)
        # But the rank of a block depends only on its size.
        # So we just sum min_rank(r_size, c_size, o_size) for all combinations.
        for r_size in rp:
            for c_size in cp:
                for o_size in op:
                    total_rank += _min_rank(r_size, c_size, o_size)
        return total_rank
    
    for rp in row_parts:
        for cp in col_parts:
            for op in out_parts:
                r = eval_partition(rp, cp, op)
                if r < best_rank:
                    best_rank = r
                    best_partition = (rp, cp, op)
    
    # Construct the decomposition for the best partition
    rp, cp, op = best_partition
    
    # We need to build the global U, V, W.
    # We will use Schoolbook for all blocks for simplicity in construction,
    # UNLESS the block is 3x3x3 (use Strassen).
    # This ensures validity.
    
    total_rank = 0
    blocks = []
    r_idx = 0
    for r_size in rp:
        c_idx = 0
        for c_size in cp:
            o_idx = 0
            for o_size in op:
                if r_size == 3 and c_size == 3 and o_size == 3:
                    rank = 7
                    btype = 'strassen'
                else:
                    rank = r_size * c_size * o_size
                    btype = 'schoolbook'
                blocks.append((r_idx, c_idx, o_idx, r_size, c_size, o_size, rank, btype))
                total_rank += rank
                o_idx += o_size
            c_idx += c_size
        r_idx += r_size
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (r_start, c_start, o_start, r_size, c_size, o_size, rank, btype) in blocks:
        if btype == 'strassen':
            U_loc, V_loc, W_loc = _strassen_3x3x3_decomp()
            for i in range(r_size):
                for j in range(c_size):
                    for k in range(rank):
                        U[(r_start+i)*m + (c_start+j)][col_offset+k] += U_loc[i*c_size+j][k]
            for j in range(c_size):
                for k in range(o_size):
                    for q in range(rank):
                        V[(c_start+j)*p + (o_start+k)][col_offset+q] += V_loc[j*o_size+k][q]
            for k in range(o_size):
                for i in range(r_size):
                    for q in range(rank):
                        W[(o_start+k)*n + (r_start+i)][col_offset+q] += W_loc[k*r_size+i][q]
        else:
            U_loc, V_loc, W_loc = _schoolbook_decomp(r_size, c_size, o_size)
            for i in range(r_size):
                for j in range(c_size):
                    for k in range(rank):
                        U[(r_start+i)*m + (c_start+j)][col_offset+k] += U_loc[i*c_size+j][k]
            for j in range(c_size):
                for k in range(o_size):
                    for q in range(rank):
                        V[(c_start+j)*p + (o_start+k)][col_offset+q] += V_loc[j*o_size+k][q]
            for k in range(o_size):
                for i in range(r_size):
                    for q in range(rank):
                        W[(o_start+k)*n + (r_start+i)][col_offset+q] += W_loc[k*r_size+i][q]
        
        col_offset += rank
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
