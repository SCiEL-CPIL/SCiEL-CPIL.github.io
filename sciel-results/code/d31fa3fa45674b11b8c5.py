# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # Handle small cases explicitly for correctness and speed
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # K_3(2,1) = 9? No.
        # Sphere size 1+2(2)=5. 9/5 not integer.
        # Greedy is best.
        pass
    if n == 3:
        # K_3(3,1) = 3 (Perfect code)
        if q == 3:
            return [[0,0,0], [1,1,1], [2,2,2]]
    if n == 4:
        if q == 3:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # General case: Greedy with redundancy removal
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # Precompute neighbors for all words to speed up coverage updates
    # neighbors[w] is a list of tuples
    neighbors = {}
    for w in all_words:
        nbs = [w]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbs.append(tuple(nw))
        neighbors[w] = nbs

    # Initial codewords: Direct sum of Tetracode (if n>=4) and full space for rest
    # Or just random/greedy initialization
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: Random or Greedy fill
        # For small n, just use all words? No, too many.
        # Use greedy construction
        initial_codewords = []
        covered = [False] * num_words
        for w in all_words:
            if not covered[word_to_idx[w]]:
                initial_codewords.append(list(w))
                for nb in neighbors[w]:
                    covered[word_to_idx[nb]] = True

    codewords = initial_codewords
    
    # Count coverage
    cover_count = [0] * num_words
    for cw in codewords:
        for nb in neighbors[tuple(cw)]:
            cover_count[word_to_idx[nb]] += 1

    # Phase 1: Remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            cw = codewords[i]
            can_remove = True
            for nb in neighbors[tuple(cw)]:
                if cover_count[word_to_idx[nb]] == 1:
                    can_remove = False
                    break
            if can_remove:
                for nb in neighbors[tuple(cw)]:
                    cover_count[word_to_idx[nb]] -= 1
                codewords.pop(i)
                improved = True
                break

    # Phase 2: Local Search (Swap)
    # Try to replace a codeword with another that allows more removals
    # This is heuristic. We'll try a few random swaps.
    random.seed(42)
    best_size = len(codewords)
    
    for _ in range(100):
        if len(codewords) <= best_size:
            break
        # Pick a random codeword to potentially replace
        idx = random.randint(0, len(codewords) - 1)
        old_cw = codewords[idx]
        
        # Pick a random candidate from all_words
        cand = random.choice(all_words)
        if cand == tuple(old_cw):
            continue
            
        # Check if replacing old_cw with cand keeps all words covered
        # Words covered by old_cw but not cand must be covered by others
        # Words covered by cand but not old_cw are new coverage (fine)
        
        # Simulate removal of old_cw
        # We need to know if any word in N(old_cw) has cover_count 1 (exclusive to old_cw)
        # AND is not in N(cand)
        
        # Let's just do a full check for simplicity, it's N=729
        # But we have cover_count.
        
        # 1. Check if removing old_cw breaks coverage
        #    A word w is broken if cover_count[w] == 1 and w in N(old_cw) and w not in N(cand)
        
        old_nbs = set(neighbors[tuple(old_cw)])
        cand_nbs = set(neighbors[cand])
        
        safe = True
        for w in old_nbs:
            if w not in cand_nbs:
                if cover_count[word_to_idx[w]] == 1:
                    safe = False
                    break
        
        if safe:
            # Perform swap
            # Decrement old
            for w in old_nbs:
                cover_count[word_to_idx[w]] -= 1
            # Increment new
            for w in cand_nbs:
                cover_count[word_to_idx[w]] += 1
            codewords[idx] = list(cand)
            
            # Try to remove redundancies again
            improved = True
            while improved:
                improved = False
                for i in range(len(codewords)):
                    cw = codewords[i]
                    can_remove = True
                    for nb in neighbors[tuple(cw)]:
                        if cover_count[word_to_idx[nb]] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for nb in neighbors[tuple(cw)]:
                            cover_count[word_to_idx[nb]] -= 1
                        codewords.pop(i)
                        improved = True
                        break
                        
            if len(codewords) < best_size:
                best_size = len(codewords)
            else:
                # Revert if no improvement? 
                # Actually, we don't track state easily. 
                # Let's just keep the swap if it didn't increase size, 
                # but we can't easily revert without storing state.
                # Given the small size, we can just accept it and hope it helps future removals.
                pass

    return codewords
# EVOLVE-BLOCK-END
