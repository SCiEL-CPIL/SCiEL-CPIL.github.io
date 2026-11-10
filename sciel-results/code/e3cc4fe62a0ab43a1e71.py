# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius radius.
    Uses a greedy algorithm with local search (2-opt style swaps) to minimize code size.
    """
    # Deadline guard: return best result before 150s wall-clock
    start_time = time.time()
    DEADLINE = 140.0  # safety margin
    
    # For the specific task: q=3, n=6, radius=1
    # Total space size: q^n = 729
    # Each codeword covers: 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Lower bound: ceil(729 / 13) = 57, but best known is 73
    
    # Step 1: Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = q ** n
    
    # Map word to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighborhoods: for each word, list of words within Hamming distance <= radius
    # For radius=1, this is the word itself + all words differing in exactly 1 position
    neighborhoods = []
    for w in all_words:
        nbrs = [w]
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list[pos] = val
                nbrs.append(tuple(w_list))
            w_list[pos] = orig
        neighborhoods.append(tuple(nbrs))
    
    # Convert neighborhoods to sets of indices for fast intersection
    nbr_sets = []
    for nbrs in neighborhoods:
        idx_set = set(word_to_idx[nbr] for nbr in nbrs)
        nbr_sets.append(idx_set)
    
    # Step 2: Greedy construction - pick codewords that cover the most uncovered words
    covered = [False] * total_words
    code = []
    
    # For efficiency, track uncovered count per potential codeword
    uncovered_count = [len(nbr_sets[i]) for i in range(total_words)]
    
    while not all(covered):
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            if not covered[i]:
                # Count how many uncovered words this would cover
                cnt = sum(1 for j in nbr_sets[i] if not covered[j])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                # Early termination if we found a perfect cover
                if best_count == len(nbr_sets[i]):
                    break
        if best_idx == -1:
            break
        # Add this codeword
        code.append(all_words[best_idx])
        for j in nbr_sets[best_idx]:
            covered[j] = True
    
    # Step 3: Local search - try to remove codewords if possible, or swap to reduce size
    # First, try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            # Temporarily remove code[i]
            code_without_i = code[:i] + code[i+1:]
            # Check if all words are still covered
            if all_words_covered(code_without_i, all_words, word_to_idx, nbr_sets, total_words):
                code = code_without_i
                changed = True
                break
    
    # Step 4: Try 2-opt swaps: replace one codeword with another to enable removal
    for _ in range(50):  # Limit iterations
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for i in range(len(code)):
            if time.time() - start_time > DEADLINE:
                break
            # Try replacing code[i] with each other word
            for j in range(len(code)):
                if i == j:
                    continue
                # Try removing code[i] and adding a new word
                code_test = code[:i] + code[i+1:]
                # Find best new word to add
                covered_test = [False] * total_words
                for cw in code_test:
                    idx = word_to_idx[cw]
                    for k in nbr_sets[idx]:
                        covered_test[k] = True
                # Find words not covered
                uncovered = [k for k in range(total_words) if not covered_test[k]]
                if not uncovered:
                    # All covered, so we can just remove code[i]
                    code = code_test
                    improved = True
                    break
                # Find the word that covers the most uncovered
                best_new = -1
                best_covers = 0
                for u in uncovered[:min(20, len(uncovered))]:  # Sample for speed
                    cnt = sum(1 for k in nbr_sets[u] if not covered_test[k])
                    if cnt > best_covers:
                        best_covers = cnt
                        best_new = u
                if best_new != -1 and best_covers > 0:
                    new_code = code_test + [all_words[best_new]]
                    # Check if this new code is valid
                    if all_words_covered(new_code, all_words, word_to_idx, nbr_sets, total_words):
                        code = new_code
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return code

def all_words_covered(code, all_words, word_to_idx, nbr_sets, total_words):
    """Check if all words are covered by the given code."""
    covered = [False] * total_words
    for cw in code:
        idx = word_to_idx[cw]
        for k in nbr_sets[idx]:
            covered[k] = True
    return all(covered)
# EVOLVE-BLOCK-END
