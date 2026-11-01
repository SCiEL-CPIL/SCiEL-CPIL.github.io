import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct(q, n, radius):
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map words to indices
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build the coverage matrix A
    # A[i, j] = 1 if word i is within radius of codeword candidate j
    # Here, candidates are all possible words, so j corresponds to all_words[j]
    # We want to select a subset of these candidates.
    
    # For small n, this matrix is dense but small.
    # For n=6, q=3, size is 729x729.
    
    # Precompute distances or coverage
    # To save memory and time, we can compute the rows of A on the fly or store as sparse.
    # SciPy milp accepts sparse matrices.
    from scipy.sparse import csr_matrix
    
    # Let's build the sparse matrix
    rows = []
    cols = []
    data = []
    
    # For each word i (constraint), find all codewords j that cover it.
    # d(i, j) <= radius
    # We can iterate through all pairs, but 729^2 is ~500k, which is fine.
    
    for i, w_i in enumerate(all_words):
        for j, w_j in enumerate(all_words):
            # Calculate Hamming distance
            dist = sum(1 for a, b in zip(w_i, w_j) if a != b)
            if dist <= radius:
                rows.append(i)
                cols.append(j)
                data.append(1)
                
    A = csr_matrix((data, (rows, cols)), shape=(num_words, num_words))
    
    # Objective: minimize sum x_j
    c = np.ones(num_words)
    
    # Constraints: A x >= 1
    # milp requires constraints in the form lb <= A_ub x <= ub
    # We can use LinearConstraint(A, 1, np.inf)
    cons = LinearConstraint(A, 1, np.inf)
    
    # Bounds: x_j in {0, 1}
    bounds = Bounds(0, 1, integer=np.ones(num_words, dtype=bool))
    
    try:
        res = milp(c, constraints=cons, bounds=bounds)
        if res.status == 0:
            # Extract codewords
            selected = [i for i in range(num_words) if res.x[i] > 0.5]
            code = [list(all_words[i]) for i in selected]
            return code
        else:
            # Fallback to greedy if ILP fails
            pass
    except Exception:
        pass
        
    # Fallback: Greedy Set Cover
    uncovered = set(range(num_words))
    code_indices = []
    
    # Precompute coverage sets for speed
    # cover[j] = set of i such that d(i, j) <= radius
    cover = [set() for _ in range(num_words)]
    for i in range(num_words):
        for j in range(num_words):
            w_i = all_words[i]
            w_j = all_words[j]
            dist = sum(1 for a, b in zip(w_i, w_j) if a != b)
            if dist <= radius:
                cover[j].add(i)
                
    while uncovered:
        # Find j that covers max uncovered
        best_j = -1
        max_cov = -1
        for j in range(num_words):
            if j in code_indices:
                continue
            cov_count = len(uncovered & cover[j])
            if cov_count > max_cov:
                max_cov = cov_count
                best_j = j
                
        if best_j == -1:
            break
            
        code_indices.append(best_j)
        uncovered -= cover[best_j]
        
    code = [list(all_words[i]) for i in code_indices]
    return code
