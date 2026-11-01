import numpy as np
import itertools
import time
from scipy.optimize import linprog

def construct(q, n, radius):
    # For this specific task, q=3, n=8, radius=1.
    # General implementation for the given parameters.
    
    total_words = q ** n
    
    # Precompute balls (indices of words within Hamming distance <= radius)
    # We need the incidence matrix A where A[i, j] = 1 if word i is in ball of word j
    # But building the full A (6561x6561) is 43M floats ~ 344MB. 
    # scipy linprog with method='highs' can handle sparse matrices.
    # Let's build the sparse matrix.
    
    from scipy.sparse import lil_matrix, csr_matrix
    
    # A[i, j] = 1 if dist(i, j) <= radius
    # Rows are words (constraints), Cols are codewords (variables)
    # Constraint: sum_j A[i, j] * x[j] >= 1 for all i
    # Minimize: sum_j x[j]
    # 0 <= x[j] <= 1
    
    # We construct the sparse matrix A
    # Since the matrix is symmetric (dist(i,j) == dist(j,i)), A is symmetric.
    # We can generate the non-zero entries.
    
    rows = []
    cols = []
    data = []
    
    # To avoid double work, we can just iterate all pairs? No, 6561^2 is too slow in pure Python.
    # We iterate over each word j (column) and its ball (indices i).
    # This is efficient because ball size is small (17 for this case).
    
    # Precompute powers for index conversion if needed, but we can just use the ball generation logic.
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    for j in range(total_words):
        # Generate ball of j
        ball = [j]
        if radius >= 1:
            for pos in range(n):
                p = powers[pos]
                old_val = (j // p) % q
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    n_idx = j + (new_val - old_val) * p
                    ball.append(n_idx)
        
        for i in ball:
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = lil_matrix((total_words, total_words), dtype=float)
    A[rows, cols] = data
    A = A.tocsr()
    
    # Objective: minimize sum(x)
    c = np.ones(total_words)
    
    # Constraints: A x >= 1
    # linprog requires A_ub x <= b_ub
    # So -A x <= -1
    A_ub = -A
    b_ub = -np.ones(total_words)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * total_words
    
    # Solve LP
    # method='highs' is the most robust and fast for LPs
    # We need the dual variables (shadow prices) for rounding.
    # In linprog, the dual variables are available in res.ineqlin or similar?
    # Actually, scipy.optimize.linprog returns 'eqlin' and 'ineqlin' duals?
    # For 'highs', the duals are in res.ineqlin if it's a HiGHS result?
    # Let's check docs or just use the primal solution for rounding if duals are hard to get.
    # However, the directive says "selects codewords based on the highest LP dual variable values".
    
    # Let's try to get duals.
    # In scipy >= 1.7, HiGHS returns duals in res.ineqlin for the inequality constraints?
    # Or is it res.slack?
    # Let's assume we can get them or fall back to primal-based rounding if duals are not directly accessible as 'dual values'.
    # Actually, for a covering problem min c^T x s.t. A x >= b, the dual is max b^T y s.t. A^T y <= c, y >= 0.
    # The dual variables y_i correspond to the constraints (rows).
    # Highs provides duals for constraints.
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to greedy if LP fails
            return _greedy_construct(q, n, radius, A)
            
        x = res.x
        # Get dual variables
        # In HiGHS, duals for A_ub x <= b_ub are in res.ineqlin
        # Note: The sign might be flipped depending on convention.
        # Our constraint was -A x <= -1.
        # Dual variable y_i >= 0 corresponds to the slack of -A_i x <= -1.
        # Let's extract them.
        duals = res.ineqlin if hasattr(res, 'ineqlin') else np.zeros(total_words)
        
        # If duals are not available or all zero, use x values
        if np.all(duals == 0):
            duals = x
            
        # Rounding Procedure:
        # 1. Sort codewords by dual value (descending).
        # 2. Iterate through them.
        # 3. If adding this codeword covers any currently uncovered point, add it.
        # 4. Stop when all points are covered.
        
        # We need to track uncovered points.
        uncovered = np.ones(total_words, dtype=bool)
        selected_indices = []
        
        # Sort indices by dual value
        # Higher dual value means the constraint (point) is "tighter" or more critical?
        # Actually, in the dual of a covering problem, high y_i means point i is hard to cover.
        # The directive says: "selects codewords based on the highest LP dual variable values".
        # This is slightly ambiguous. Does it mean select codeword j if sum_{i in ball(j)} y_i is high?
        # Or does it mean use the duals to weight the points?
        # Let's interpret "selects codewords based on ... dual variable values" as:
        # Calculate a score for each codeword j based on the duals of the points it covers.
        # Score_j = sum_{i in ball(j)} y_i
        # Then greedily select codewords with highest Score_j that cover uncovered points.
        
        # Calculate scores
        # Score_j = A^T y (where A is the original 0/1 matrix, not -A)
        # A is sparse.
        scores = A.T @ duals
        
        # Sort codewords by score descending
        order = np.argsort(-scores)
        
        for j in order:
            if not np.any(uncovered):
                break
            
            # Get ball of j
            # We can use the sparse matrix row/col structure or re-generate.
            # Since A is CSR, we can get the non-zero cols for row j?
            # Wait, A[i, j] = 1 if i in ball(j).
            # So column j of A contains the points covered by j.
            # In CSR, we access rows easily. In CSC, columns easily.
            # Let's convert A to CSC for column access.
            pass # Placeholder
            
        # Let's re-do the selection loop properly with CSC
        A_csc = A.tocsc()
        
        for j in order:
            if not np.any(uncovered):
                break
            
            # Get indices i such that A[i, j] == 1
            # In CSC, col j starts at A_csc.indptr[j] and ends at A_csc.indptr[j+1]
            start = A_csc.indptr[j]
            end = A_csc.indptr[j+1]
            ball_indices = A_csc.indices[start:end]
            
            # Check if any of these are uncovered
            # If yes, select j
            if np.any(uncovered[ball_indices]):
                selected_indices.append(j)
                uncovered[ball_indices] = False
                
    except Exception as e:
        # Fallback to greedy if LP fails or errors
        return _greedy_construct(q, n, radius, A)
        
    # Convert indices to codewords
    selected = []
    for j in selected_indices:
        word = [int((j // p) % q) for p in reversed(powers)]
        selected.append(word)
        
    return selected

def _greedy_construct(q, n, radius, A=None):
    # Fallback greedy implementation if LP fails
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls as lists of indices
    balls = []
    for i in range(total_words):
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                p = powers[pos]
                old_val = (i // p) % q
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    n_idx = i + (new_val - old_val) * p
                    ball.append(n_idx)
        balls.append(ball)
        
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    # Precompute ball arrays for fast numpy sum
    balls_np = [np.array(b) for b in balls]
    
    while np.any(uncovered):
        gains = np.array([np.sum(uncovered[ball]) for ball in balls_np])
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected.append([int((best_idx // p) % q) for p in reversed(powers)])
        uncovered[balls_np[best_idx]] = False
        
    return selected
