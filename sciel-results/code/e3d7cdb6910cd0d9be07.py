import itertools
import random
import numpy as np
import scipy.sparse as sp
import scipy.optimize

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Fallback Greedy Method
    def greedy_fallback():
        # Precompute spheres
        spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            base_digits = index_to_digits(i)
            sphere = set()
            for r in range(radius + 1):
                for pos_subset in itertools.combinations(range(n), r):
                    if r == 0:
                        sphere.add(i)
                        continue
                    for vals in itertools.product(range(q), repeat=r):
                        new_digits = base_digits[:]
                        for k, pos in enumerate(pos_subset):
                            new_digits[pos] = vals[k]
                        new_idx = digits_to_index(new_digits)
                        sphere.add(new_idx)
            spheres[i] = list(sphere)
        
        covered = [False] * total_words
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                new_count = sum(1 for w in spheres[i] if not covered[w])
                if new_count > best_count:
                    best_count = new_count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            codewords.append(best_idx)
            for w in spheres[best_idx]:
                if not covered[w]:
                    covered[w] = True
                    remaining -= 1
        
        return [index_to_digits(idx) for idx in codewords]

    # Try ILP approach
    if hasattr(scipy.optimize, 'milp'):
        try:
            rows = []
            cols = []
            data = []
            
            for j in range(total_words):
                base_digits = index_to_digits(j)
                neighbors = set()
                neighbors.add(j)
                
                for r in range(1, radius + 1):
                    for pos_subset in itertools.combinations(range(n), r):
                        for vals in itertools.product(range(q), repeat=r):
                            new_digits = base_digits[:]
                            for k, pos in enumerate(pos_subset):
                                new_digits[pos] = vals[k]
                            new_idx = digits_to_index(new_digits)
                            neighbors.add(new_idx)
                
                for c in neighbors:
                    rows.append(j)
                    cols.append(c)
                    data.append(1)

            A = sp.csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
            
            c_obj = np.ones(total_words)
            cons = scipy.optimize.LinearConstraint(A, 1, np.inf)
            integrality = np.ones(total_words)
            bounds = scipy.optimize.Bounds(lb=0, ub=1)
            
            res = scipy.optimize.milp(c=c_obj, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.success:
                x = res.x
                selected_indices = np.where(x > 0.5)[0]
                return [index_to_digits(int(idx)) for idx in selected_indices]
            else:
                # If ILP fails, fall back
                return greedy_fallback()
                
        except Exception:
            return greedy_fallback()
    else:
        return greedy_fallback()
