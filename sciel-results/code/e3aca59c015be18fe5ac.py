import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct(q, n, radius):
        # Parameters for the specific task
        # q=3, n=8, radius=1
        # Total words
        N = q ** n
        
        # Precompute neighbors (words within distance radius)
        # For each word c, list of words w such that d(c,w) <= radius
        # We need the constraint matrix A: A[w, c] = 1 if c covers w
        
        # To save memory and time, we can build the sparse matrix directly.
        # But we need to map words to indices.
        
        # Helper: convert base-q tuple to index
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n-1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        
        # Build list of all words
        all_words = [idx_to_word(i) for i in range(N)]
        
        # Build sparse matrix A
        # Rows: w (universe), Cols: c (candidate codewords)
        # A[w, c] = 1 if dist(w, c) <= radius
        
        rows = []
        cols = []
        data = []
        
        # Iterate over all candidate codewords c
        for c_idx in range(N):
            c_word = all_words[c_idx]
            # Find all w within distance radius
            # For radius 1, this is c itself + neighbors with 1 diff
            # General case:
            # We can generate neighbors
            # For efficiency, since N is small (6561), we can just check all pairs? 
            # 6561^2 is 43M. In Python, this loop might be slow (seconds to minutes).
            # Better: Generate neighbors of c.
            
            # Generate all words within distance radius
            # Use itertools.product for variations?
            # For radius 1:
            # c
            # c with one position changed
            
            covered = set()
            # Add c
            covered.add(c_idx)
            
            # Add neighbors
            for i in range(n):
                original_val = c_word[i]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_word = c_word.copy()
                    new_word[i] = new_val
                    w_idx = word_to_idx(new_word)
                    covered.add(w_idx)
            
            # For each w in covered, A[w, c_idx] = 1
            for w_idx in covered:
                rows.append(w_idx)
                cols.append(c_idx)
                data.append(1)
                
        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        
        # Objective: minimize sum x
        c_obj = np.ones(N)
        
        # Constraints: A x >= 1
        # milp expects LinearConstraint(A, lb, ub)
        # lb = 1, ub = inf
        constraint = LinearConstraint(A, 1, np.inf)
        
        # Bounds: x in [0, 1]
        bounds = Bounds(0, 1)
        
        # Integrality: all binary
        integrality = np.ones(N)
        
        start_time = time.time()
        timeout = 100 # Leave margin for 120s limit
        
        try:
            res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options={"time_limit": timeout})
            
            if res.status == 0: # Optimal
                x = res.x
                code_indices = [i for i in range(N) if x[i] > 0.5]
            elif res.status == 1: # Feasible (not optimal, e.g. time limit)
                x = res.x
                code_indices = [i for i in range(N) if x[i] > 0.5]
            else:
                # Fallback to tetracode
                code_indices = []
                
        except Exception as e:
            # Fallback to tetracode
            code_indices = []
            
        if not code_indices:
            # Tetracode fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            return code
        else:
            code = [all_words[i] for i in code_indices]
            return code
