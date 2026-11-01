"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (evaluator kills at 150s)

    # Base construction: direct sum of ternary tetracode (9 words of length 4)
    # with all words on the remaining n-4 coordinates.
    # Size: 9 * 3**(n-4)
    tetracode = _tetracode()
    if n == 4:
        base = [list(w) for w in tetracode]
    else:
        base = [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    # Try to reduce via greedy removal / repair
    # For n=8, radius=1, base size = 9 * 3^4 = 6561... wait that's too big.
    # Actually 9 * 81 = 729.
    # The known optimum is 486.
    
    # Let's think about this differently.
    # The tetracode approach gives 9 * 3^(n-4). For n=8: 9*81 = 729.
    # We need to get down to ~486.
    
    # Strategy: multi-start stochastic local search with delete-and-repair.
    # 1. Start with the 729-word code.
    # 2. Repeatedly try to remove codewords that are redundant.
    # 3. If removal creates uncovered words, try to repair by adding a better word.
    
    # But first, let's use a smarter initial construction.
    # For q=3, n=8, radius=1:
    # The ternary tetracode is a perfect [4,2,3] code with covering radius 1.
    # Direct product with full space on 4 coords gives 9*81=729.
    
    # Alternative: use a greedy set cover approach.
    # Each codeword covers (1 + n*(q-1)) words = 1 + 8*2 = 17 words.
    # Total words: 3^8 = 6561.
    # Lower bound: 6561/17 ≈ 386.
    # Known optimum: 486.
    
    # Let's implement a greedy + local search approach.
    
    # For efficiency, represent the space as integers 0..6560.
    # A word is covered if it's within Hamming distance 1 of some codeword.
    
    # Precompute: for each word w, the set of words within distance 1.
    # This is 6561 * 17 entries.
    
    # Greedy set cover:
    # 1. Start with all words uncovered.
    # 2. Repeatedly pick the codeword that covers the most uncovered words.
    # 3. Mark those words as covered.
    # 4. Stop when all are covered.
    
    # Then local search: try to remove codewords and repair.
    
    # For n=8, q=3, radius=1, this should be feasible within time.
    
    if n != 8 or q != 3 or radius != 1:
        # Fallback: use the tetracode construction for general case
        return base

    # Specialized path for q=3, n=8, radius=1
    total_words = 3**8
    N = total_words  # 6561
    
    # Convert between word representation (list of 8 ints) and integer index
    def word_to_idx(w):
        idx = 0
        for i in range(8):
            idx = idx * 3 + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0]*8
        for i in range(7, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w
    
    # Precompute neighborhood: for each word, list of words within Hamming distance 1
    # neighborhood[i] = list of indices within distance 1 of word i
    neighborhood = []
    for idx in range(N):
        w = idx_to_word(idx)
        nb = [idx]
        for pos in range(8):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    nb.append(word_to_idx(w))
                    w[pos] = orig
        neighborhood.append(nb)
    
    # Greedy set cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # For greedy: pick codeword covering most uncovered words
    # To speed up, we can use a priority queue, but for N=6561 and ~486 iterations,
    # a simple approach might work.
    
    # Actually, let's use a different strategy:
    # Start with the tetracode-based code (729 words), then do local search to reduce.
    
    # Convert base code to indices
    selected_indices = set(word_to_idx(w) for w in base)
    
    # Compute coverage: for each word, how many selected codewords cover it
    # cover_count[i] = number of selected codewords c such that i is in neighborhood[c]
    cover_count = np.zeros(N, dtype=np.int32)
    for c_idx in selected_indices:
        for nb_idx in neighborhood[c_idx]:
            cover_count[nb_idx] += 1
    
    # Local search: try to remove codewords
    # A codeword c can be removed if for all i in neighborhood[c], cover_count[i] >= 2
    # (i.e., i is covered by at least one other codeword)
    
    # We'll do iterative passes:
    # 1. Find all removable codewords.
    # 2. Remove them.
    # 3. Repeat until no more can be removed.
    
    # Then try "swap" moves: remove a codeword, and if some words become uncovered,
    # add a different codeword that covers those uncovered words.
    
    # For efficiency, let's track which words are uniquely covered by each codeword.
    
    # Let's implement a simpler but effective local search:
    
    def try_remove_and_repair(selected_set, cover_count):
        """Try to remove codewords and repair if needed."""
        removed_any = False
        # Try each codeword in random order
        candidates = list(selected_set)
        random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > deadline - 5.0:
                break
            if c_idx not in selected_set:
                continue
            
            # Check if c_idx is removable
            unique_words = []
            for nb_idx in neighborhood[c_idx]:
                if cover_count[nb_idx] == 1:
                    unique_words.append(nb_idx)
            
            if not unique_words:
                # Can remove without repair
                for nb_idx in neighborhood[c_idx]:
                    cover_count[nb_idx] -= 1
                selected_set.remove(c_idx)
                removed_any = True
            else:
                # Try to find a replacement codeword that covers all unique_words
                # and is not already selected
                # The replacement must cover all words in unique_words
                # and should ideally cover many other words too.
                
                # Find candidate replacements: words that cover all of unique_words
                # A word x covers y if y is in neighborhood[x], i.e., x is in neighborhood[y]
                
                # For each unique word u, the set of words that cover u is neighborhood[u]
                # We need x such that x is in neighborhood[u] for all u in unique_words
                # i.e., x is in the intersection of neighborhood[u] for all u in unique_words
                
                if len(unique_words) > 3:
                    # Too many constraints, skip
                    continue
                
                # Find intersection
                candidate_set = set(neighborhood[unique_words[0]])
                for u in unique_words[1:]:
                    candidate_set.intersection_update(neighborhood[u])
                
                # Filter out already selected words
                candidate_set -= selected_set
                
                if not candidate_set:
                    continue
                
                # Pick the candidate that maximizes coverage of currently uncovered words
                # (after removing c_idx, the unique_words become uncovered)
                # Actually, after removing c_idx, only unique_words become uncovered.
                # We want a replacement that covers unique_words and maximizes additional coverage.
                
                best_candidate = None
                best_score = -1
                for x in candidate_set:
                    # Score: number of words in neighborhood[x] that are currently covered
                    # by at least one other selected word (i.e., cover_count >= 2 after removal)
                    # This is an approximation; let's just pick the one with largest neighborhood
                    # overlap with currently well-covered words.
                    # Simpler: just pick the first one (random order already applied)
                    best_candidate = x
                    break
                
                if best_candidate is not None:
                    # Remove c_idx
                    for nb_idx in neighborhood[c_idx]:
                        cover_count[nb_idx] -= 1
                    selected_set.remove(c_idx)
                    
                    # Add best_candidate
                    for nb_idx in neighborhood[best_candidate]:
                        cover_count[nb_idx] += 1
                    selected_set.add(best_candidate)
                    removed_any = True
        
        return removed_any
    
    # Iterative removal
    selected_set = set(selected_indices)
    
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        if not try_remove_and_repair(selected_set, cover_count):
            break
    
    # Convert back to list of words
    result = [idx_to_word(idx) for idx in selected_set]
    
    # Verify coverage (quick check)
    # For safety, verify that all words are covered
    covered = np.zeros(N, dtype=bool)
    for c_idx in selected_set:
        for nb_idx in neighborhood[c_idx]:
            covered[nb_idx] = True
    
    if not np.all(covered):
        # Fallback to base
        return base
    
    return result
# EVOLVE-BLOCK-END
