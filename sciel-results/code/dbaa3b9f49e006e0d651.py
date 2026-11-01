"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize

def construct(q, n, radius):
    """
    Construct a covering code using a density optimization approach.
    We model the problem as finding a probability distribution over words
    that minimizes the expected number of codewords while ensuring coverage.
    We use gradient descent on a soft penalty function (relaxation) and then
    round the solution to a discrete set using a greedy strategy guided by
    the learned probabilities.
    """
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Build adjacency matrix A where A[i, j] = 1 if dist(i, j) <= radius
        # A is symmetric.
        A = np.zeros((total_words, total_words), dtype=np.float64)
        for i, w in enumerate(all_words):
            # Self
            A[i, i] = 1.0
            # Neighbors
            for pos in range(n):
                for val in range(3):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = word_to_idx[tuple(new_w)]
                    A[i, j] = 1.0
        
        # Initialize probabilities p
        # Start with uniform distribution scaled to roughly meet coverage
        # Each word covers 1 + n*(q-1) = 1 + 6*2 = 13 words.
        # To cover 729 words, we need approx 729/13 ~ 56 words.
        # So average p ~ 56/729 ~ 0.077.
        p = np.full(total_words, 0.05)
        
        # Gradient Descent
        # Loss L(p) = sum(p) + lambda * sum(relu(1 - C_i)^2)
        # C_i = sum_j A[i,j] p_j = (A @ p)_i
        # Grad L_k = 1 + 2 * lambda * sum_{i: k in N(i) and C_i < 1} (1 - C_i)
        # Note: k in N(i) <=> i in N(k) due to symmetry of Hamming dist.
        
        lr = 0.01
        lam = 10.0
        steps = 500
        
        for step in range(steps):
            C = A @ p
            # Deficit
            deficit = np.maximum(0, 1.0 - C)
            
            # Gradient
            # grad = 1 + 2 * lam * A.T @ deficit
            grad = 1.0 + 2.0 * lam * (A.T @ deficit)
            
            # Update
            p -= lr * grad
            # Clip to [0, 1]
            p = np.clip(p, 0.0, 1.0)
            
            # Optional: decay learning rate
            lr *= 0.99
            
        # Rounding Strategy:
        # Sort indices by p descending
        indices = np.argsort(p)[::-1]
        
        code_set = set()
        covered_mask = np.zeros(total_words, dtype=bool)
        
        for idx in indices:
            if covered_mask.all():
                break
            # Add this word
            code_set.add(all_words[idx])
            # Update coverage
            # covered_mask |= A[idx, :] > 0
            # Since A[idx, :] has 1s for neighbors
            covered_mask |= (A[idx, :] > 0)
            
        # Verify coverage
        if not covered_mask.all():
             # Fallback: add greedy words for any remaining uncovered
             uncovered_indices = np.where(~covered_mask)[0]
             for u_idx in uncovered_indices:
                 # Find best word to cover u_idx
                 # Just pick the one with highest p that covers it
                 candidates = np.where(A[u_idx, :] > 0)[0]
                 if len(candidates) > 0:
                     best_cand = candidates[np.argmax(p[candidates])]
                     code_set.add(all_words[best_cand])
                     covered_mask |= (A[best_cand, :] > 0)
        
        result = [list(c) for c in code_set]
        return result

    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
