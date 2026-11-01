import numpy as np
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    # For this task, q=3, n=8, radius=1.
    # Generate all words in the space
    total_words = q ** n
    
    # Precompute powers of q for index arithmetic
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)
    
    # Build coverage masks: masks[i] is a boolean array where masks[i][j]=True 
    # if word i covers word j (Hamming distance <= radius)
    # For efficiency, we'll build sparse representation or use the mask directly
    
    # For q=3, n=8, total_words=6561
    # Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words for radius=1
    # Total non-zeros in mask: 6561 * 17 ≈ 111,537
    
    # Build coverage list for each word
    covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        covers[i].append(i)  # self
        if radius >= 1:
            for pos in range(n):
                orig = all_words[i][pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    n_idx = i + (new_val - orig) * powers[pos]
                    covers[i].append(n_idx)
    
    # Set up LP: minimize sum(x_i) subject to for each word j, sum_{i: i covers j} x_i >= 1
    # x_i in [0, 1]
    
    # c = [1, 1, ..., 1] (minimize number of codewords)
    c = np.ones(total_words)
    
    # A_ub: for each word j, we need sum of x_i for i covering j >= 1
    # linprog uses A_ub @ x <= b_ub, so we negate: -sum <= -1
    # Build A_ub as sparse-like structure using dense for this size
    
    # A_ub[j, i] = -1 if codeword i covers word j, else 0
    # This is the transpose of the coverage matrix, negated
    
    # Build A_ub efficiently
    A_ub = np.zeros((total_words, total_words), dtype=np.float64)
    for i in range(total_words):
        for j in covers[i]:
            A_ub[j, i] = -1.0
    
    b_ub = -np.ones(total_words)
    
    # Bounds: x_i in [0, 1]
    bounds = [(0, 1)] * total_words
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x
    else:
        # Fallback to greedy if LP fails
        x = None
    
    if x is None or not np.all(x >= -1e-9):
        # Fallback to greedy set cover
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        while np.any(uncovered):
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if uncovered[i]:
                    gain = sum(1 for j in covers[i] if uncovered[j])
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(all_words[best_idx].tolist())
            for j in covers[best_idx]:
                uncovered[j] = False
        return selected
    
    # Rounding procedure: select codewords with highest LP values
    # Sort by LP value descending, greedily add until all covered
    order = np.argsort(-x)  # descending order of x values
    
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    for i in order:
        if not np.any(uncovered):
            break
        # Check if this codeword covers any uncovered word
        gain = sum(1 for j in covers[i] if uncovered[j])
        if gain > 0:
            selected.append(all_words[i].tolist())
            for j in covers[i]:
                uncovered[j] = False
    
    # If not all covered, add remaining uncovered words as codewords (shouldn't happen)
    if np.any(uncovered):
        for j in range(total_words):
            if uncovered[j]:
                selected.append(all_words[j].tolist())
                uncovered[j] = False
                for k in covers[j]:
                    uncovered[k] = False
    
    return selected
