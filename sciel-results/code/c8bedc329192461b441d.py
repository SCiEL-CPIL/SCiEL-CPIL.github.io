import numpy as np
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    # For this specific task, q=3, n=8, radius=1.
    # Use LP relaxation + dual-variable rounding for set cover.
    
    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    total_words = len(all_words)
    
    # Precompute coverage masks: masks[i, j] = True if codeword i covers word j
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Build adjacency: for each word, list of words it covers (itself + neighbors within radius)
    coverage = []
    for i in range(total_words):
        word = all_words[i]
        covered = [i]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    n_idx = i + (new_val - orig) * powers[pos]
                    covered.append(n_idx)
        coverage.append(covered)
    
    # Build the constraint matrix A where A[j, i] = 1 if codeword i covers word j
    # We need: sum_i A[j,i] * x[i] >= 1 for all j
    # In linprog form: -A @ x <= -1, x >= 0, minimize sum(x)
    
    # A has shape (total_words, total_words)
    A = np.zeros((total_words, total_words), dtype=np.float64)
    for i in range(total_words):
        for j in coverage[i]:
            A[j, i] = 1.0
    
    # Objective: minimize sum(x_i)
    c = np.ones(total_words, dtype=np.float64)
    
    # Constraints: -A @ x <= -1
    A_ub = -A
    b_ub = -np.ones(total_words, dtype=np.float64)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * total_words
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback to greedy if LP fails
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        while np.any(uncovered):
            gains = np.sum((A.T > 0) & uncovered[:, None], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(all_words[best_idx].tolist())
            uncovered[coverage[best_idx]] = False
        return selected
    
    x_lp = result.x
    
    # Get dual variables from the LP solution
    # The dual variables correspond to the constraints -A @ x <= -1
    # In HiGHS, the dual variables are in result.ineqlin.marginals
    duals = result.ineqlin.marginals if hasattr(result, 'ineqlin') else None
    
    if duals is None or len(duals) == 0:
        # Fallback: use primal solution to rank codewords
        # Select codewords with highest x_lp values first, then fill gaps
        order = np.argsort(-x_lp)
        selected = []
        uncovered = np.ones(total_words, dtype=bool)
        
        for idx in order:
            if not np.any(uncovered):
                break
            # Check if this codeword covers any uncovered word
            covers_uncovered = np.any(uncovered[coverage[idx]])
            if covers_uncovered:
                selected.append(all_words[idx].tolist())
                uncovered[coverage[idx]] = False
        
        # If still uncovered, use greedy to fill remaining
        if np.any(uncovered):
            while np.any(uncovered):
                gains = np.sum((A.T > 0) & uncovered[:, None], axis=1)
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                selected.append(all_words[best_idx].tolist())
                uncovered[coverage[best_idx]] = False
        
        return selected
    
    # Dual variables: duals[j] is the shadow price for constraint j
    # We want to select codewords that are "most needed" based on dual values
    # A codeword i is valuable if it covers words with high dual values
    
    # Compute value of each codeword based on duals of words it covers
    codeword_values = np.zeros(total_words, dtype=np.float64)
    for i in range(total_words):
        codeword_values[i] = np.sum(duals[coverage[i]])
    
    # Select codewords in order of decreasing value
    order = np.argsort(-codeword_values)
    
    selected = []
    uncovered = np.ones(total_words, dtype=bool)
    
    for idx in order:
        if not np.any(uncovered):
            break
        # Check if this codeword covers any uncovered word
        covers_uncovered = np.any(uncovered[coverage[idx]])
        if covers_uncovered:
            selected.append(all_words[idx].tolist())
            uncovered[coverage[idx]] = False
    
    # If still uncovered, use greedy to fill remaining
    if np.any(uncovered):
        while np.any(uncovered):
            gains = np.sum((A.T > 0) & uncovered[:, None], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(all_words[best_idx].tolist())
            uncovered[coverage[best_idx]] = False
    
    return selected
