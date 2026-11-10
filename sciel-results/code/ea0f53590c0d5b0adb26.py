import numpy as np
import scipy.sparse as sp

def construct(q, n, radius):
    total_words = q ** n
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_index(word):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + word[i]
        return idx

    # Fallback: Return all words (valid but large)
    def fallback():
        return [index_to_word(i) for i in range(total_words)]

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Build sparse constraint matrix A
        rows = []
        cols = []
        data = []
        
        for i in range(total_words):
            word = index_to_word(i)
            rows.append(i)
            cols.append(i)
            data.append(1)
            for pos in range(n):
                original_val = word[pos]
                for delta in range(1, q):
                    new_word = word[:]
                    new_word[pos] = (original_val + delta) % q
                    new_idx = word_to_index(new_word)
                    rows.append(i)
                    cols.append(new_idx)
                    data.append(1)

        A_sparse = sp.coo_matrix((data, (rows, cols)), shape=(total_words, total_words)).tocsr()
        
        c = np.ones(total_words)
        
        # Symmetry breaking: fix x_0 = 1
        A_eq = np.zeros((1, total_words))
        A_eq[0, 0] = 1.0
        b_eq = np.array([1.0])
        
        A_ub = -A_sparse.tocsr()
        b_ub = -np.ones(total_words)
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(total_words)
        
        res = milp(c, 
                   constraints=(LinearConstraint(A_ub, -np.inf, -1),
                                LinearConstraint(A_eq, b_eq, b_eq)),
                   bounds=bounds, 
                   integrality=integrality)
        
        if res.status == 0:
            x = res.x
            codewords = []
            for i in range(total_words):
                if x[i] > 0.5:
                    codewords.append(index_to_word(i))
            return codewords
        else:
            return fallback()
            
    except Exception:
        return fallback()
