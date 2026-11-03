import random
import numpy as np

def construct(q, n, radius):
    # Generate all words
    total_words = q ** n
    all_words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word, which words are within radius
    # For radius=1, a word covers itself and words differing in exactly one position
    coverage = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nw = tuple(nw)
                j = word_to_idx[nw]
                covered.append(j)
        coverage.append(covered)
    
    # Helper to check if a set of codewords covers everything
    def is_covered(codeword_set):
        # codeword_set is a set of indices
        if not codeword_set:
            return False
        covered_mask = np.zeros(total_words, dtype=bool)
        for c in codeword_set:
            covered_mask[coverage[c]] = True
        return np.all(covered_mask)
    
    # Helper to get the set of uncovered words
    def get_uncovered(codeword_set):
        covered_mask = np.zeros(total_words, dtype=bool)
        for c in codeword_set:
            covered_mask[coverage[c]] = True
        return np.where(~covered_mask)[0]

    best_solution = None
    best_size = total_words
    
    # Number of iterations to run the heuristic
    # Adjust based on time budget. 120s is plenty for a few hundred iterations.
    num_iterations = 50
    
    for _ in range(num_iterations):
        # 1. Greedy Initialization with random tie-breaking
        codewords = set()
        uncovered = set(range(total_words))
        
        while uncovered:
            best_c = -1
            best_count = -1
            best_tie = None
            # Sample a subset of candidates to speed up greedy, or use all
            # For N=2187, checking all is fine, but we can optimize.
            # Let's just check all for accuracy in greedy phase.
            for c in range(total_words):
                if c in codewords:
                    continue
                # Count how many uncovered words c covers
                # Optimization: Precomputed coverage lists
                cnt = sum(1 for j in coverage[c] if j in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    best_tie = [c]
                elif cnt == best_count and cnt > 0:
                    best_tie.append(c)
            
            if best_c == -1:
                break # Should not happen if space is finite and radius >= 0
                
            # Random tie-breaking
            chosen = random.choice(best_tie)
            codewords.add(chosen)
            # Update uncovered
            for j in coverage[chosen]:
                if j in uncovered:
                    uncovered.discard(j)
        
        # 2. Local Search: Reduction
        improved = True
        while improved:
            improved = False
            for c in list(codewords):
                # Check if c is redundant
                temp_set = codewords - {c}
                if is_covered(temp_set):
                    codewords = temp_set
                    improved = True
                    break # Restart loop as set changed
        
        # 3. Local Search: Swaps
        # Try to replace a codeword with another to potentially allow further reduction
        # This is more complex. A simple heuristic:
        # If we swap c with w, and the new set is valid, we keep it.
        # To improve size, we need the swap to enable a removal.
        # Let's try a simpler move: Random Perturbation followed by Reduction.
        
        # Alternative Local Search: First-Fit style replacement
        # Let's stick to Reduction for now, but add a "Swap" phase.
        
        # Phase 2: Try swaps to escape local minima
        for _ in range(100): # Max swap attempts
            if len(codewords) == 0:
                break
            c = random.choice(list(codewords))
            # Pick a random word not in codewords
            w = random.randint(0, total_words - 1)
            if w in codewords:
                continue
            
            new_set = (codewords - {c}) | {w}
            if is_covered(new_set):
                # If the new set is valid, does it allow reduction?
                # We can just accept it to change the state, then run reduction again.
                codewords = new_set
                # Run reduction again
                improved = True
                while improved:
                    improved = False
                    for cc in list(codewords):
                        temp_set = codewords - {cc}
                        if is_covered(temp_set):
                            codewords = temp_set
                            improved = True
                            break
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_solution = set(codewords)
            
    # Return the best solution found
    return [list(all_words[i]) for i in best_solution]
