import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Build the full space and the direct-sum tetracode construction
    # For q=3, n=6: tetracode (9 words of length 4) ⊗ all tails (9 words of length 2) = 81 words
    if n >= 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        codewords = [h + list(tail) for h in tetracode for tail in product(range(q), repeat=n - 4)]
    else:
        codewords = [list(w) for w in product(range(q), repeat=n)]

    m = len(codewords)
    total_words = q ** n

    # Map each codeword to its index in the full space
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Precompute the set of words within distance ≤ radius of each codeword
    # For radius=1, that's the word itself + n*(q-1) neighbors = 1 + 6*2 = 13 for n=6, q=3
    covers = []
    for cw in codewords:
        covered = set()
        covered.add(word_to_idx(cw))
        for pos in range(n):
            orig = cw[pos]
            for alt in range(q):
                if alt == orig:
                    continue
                cw2 = cw.copy()
                cw2[pos] = alt
                covered.add(word_to_idx(cw2))
        covers.append(list(covered))

    # Build a neighbor list: for each word, which codeword indices cover it
    # This is a 729 x 81 sparse structure. Use numpy int arrays.
    # covered_by[word_idx] = list of codeword indices covering it
    covered_by = [[] for _ in range(total_words)]
    for i in range(m):
        for w in covers[i]:
            covered_by[w].append(i)

    # Initial state: all 81 codewords active
    active = np.ones(m, dtype=bool)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for i in range(m):
        for w in covers[i]:
            cover_count[w] += 1

    def try_remove_redundant(active_arr, cover_count_arr, covers, deadline):
        """Remove redundant codewords. Returns number removed."""
        removed = 0
        order = list(range(len(active_arr)))
        random.shuffle(order)
        for i in order:
            if time.time() > deadline - 3.0:
                break
            if not active_arr[i]:
                continue
            # Check if all words covered by i have count >= 2
            redundant = True
            for w in covers[i]:
                if cover_count_arr[w] <= 1:
                    redundant = False
                    break
            if redundant:
                active_arr[i] = False
                for w in covers[i]:
                    cover_count_arr[w] -= 1
                removed += 1
        return removed

    def greedy_cover_removal(active_arr, cover_count_arr, covers, deadline):
        """Aggressively remove redundant codewords until none remain."""
        while True:
            if time.time() > deadline - 5.0:
                break
            r = try_remove_redundant(active_arr, cover_count_arr, covers, deadline)
            if r == 0:
                break

    # Phase 1: Greedy removal from the full 81-word construction
    active = np.ones(m, dtype=bool)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for i in range(m):
        for w in covers[i]:
            cover_count[w] += 1

    greedy_cover_removal(active, cover_count, covers, deadline)
    best_count = int(np.sum(active))
    best_active = active.copy()

    # Phase 2: Local search with add/remove swaps
    # We allow adding new codewords from the full space (not just the initial 81)
    # Key insight: add a random word, then remove up to 2 redundant ones
    # This is a 1-for-2 exchange
    
    # Precompute covers for ALL possible words (729 of them)
    # covers_all[w] = list of word indices within distance 1 of word w
    # Actually we already have covers for the 81 initial codewords.
    # For new words, we need to compute their covers on the fly or precompute all.
    # Precompute all 729 covers - this is cheap: 729 * 13 = ~9500 operations
    
    covers_all = []
    for idx in range(total_words):
        # Convert idx to word
        w = []
        tmp = idx
        for pos in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        covered = set()
        covered.add(idx)
        for pos in range(n):
            orig = w[pos]
            for alt in range(q):
                if alt == orig:
                    continue
                w2 = w.copy()
                w2[pos] = alt
                idx2 = 0
                for v in w2:
                    idx2 = idx2 * q + v
                covered.add(idx2)
        covers_all.append(list(covered))
    
    # Map from word index to codeword list (for the initial 81)
    word_idx_to_cw = {}
    for i, cw in enumerate(codewords):
        word_idx_to_cw[word_to_idx(cw)] = cw
    
    # For tracking, we'll use a set of active word indices
    # active_set: set of word indices that are currently in the code
    
    best_active_set = set()
    for i in range(m):
        if best_active[i]:
            best_active_set.add(word_to_idx(codewords[i]))
    
    best_count = len(best_active_set)
    
    # Build cover_count from best_active_set
    cover_count = np.zeros(total_words, dtype=np.int32)
    for wi in best_active_set:
        for w in covers_all[wi]:
            cover_count[w] += 1
    
    def compute_cover_count(active_set, deadline):
        cc = np.zeros(total_words, dtype=np.int32)
        for wi in active_set:
            for w in covers_all[wi]:
                cc[w] += 1
        return cc
    
    def try_remove_redundant_set(active_set, cover_count_arr, deadline):
        removed = 0
        candidates = list(active_set)
        random.shuffle(candidates)
        for wi in candidates:
            if time.time() > deadline - 3.0:
                break
            if wi not in active_set:
                continue
            redundant = True
            for w in covers_all[wi]:
                if cover_count_arr[w] <= 1:
                    redundant = False
                    break
            if redundant:
                active_set.discard(wi)
                for w in covers_all[wi]:
                    cover_count_arr[w] -= 1
                removed += 1
        return removed
    
    def greedy_removal_set(active_set, cover_count_arr, deadline):
        while True:
            if time.time() > deadline - 5.0:
                break
            r = try_remove_redundant_set(active_set, cover_count_arr, deadline)
            if r == 0:
                break
    
    # Phase 2: Add-then-remove local search
    # Try to add a random word, then remove redundant words. If we end up with fewer, keep it.
    
    all_word_indices = list(range(total_words))
    initial_indices = set(word_to_idx(cw) for cw in codewords)
    
    # Candidate pool: all words not in the initial set (new possibilities)
    # Actually, let's use all words as candidates
    candidate_pool = all_word_indices[:]
    random.shuffle(candidate_pool)
    
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Start from current best
        current_set = set(best_active_set)
        current_cc = compute_cover_count(current_set, deadline)
        
        # Try multiple add-remove cycles
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try to add a random word and remove redundants
            for trial in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random word not currently in the set
                if len(current_set) == total_words:
                    break
                
                # Randomly sample a few candidates
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick a random word index
                    wi = random.randint(0, total_words - 1)
                    if wi in current_set:
                        continue
                    
                    # Add it
                    current_set.add(wi)
                    for w in covers_all[wi]:
                        current_cc[w] += 1
                    
                    # Try to remove redundant words
                    greedy_removal_set(current_set, current_cc, deadline)
                    
                    if len(current_set) < best_count:
                        best_count = len(current_set)
                        best_active_set = set(current_set)
                        improved = True
                        break
                    
                    # Undo: remove the added word
                    current_set.discard(wi)
                    for w in covers_all[wi]:
                        current_cc[w] -= 1
                
                if improved:
                    # Recompute cover count from best
                    current_cc = compute_cover_count(current_set, deadline)
                    break
                
            # Also try pure removal (in case we're stuck)
            greedy_removal_set(current_set, current_cc, deadline)
            if len(current_set) < best_count:
                best_count = len(current_set)
                best_active_set = set(current_set)
                current_cc = compute_cover_count(current_set, deadline)
                improved = True
    
    # Phase 3: Swap-based local search within the solution
    # Try swapping one active word for another inactive word
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        current_set = set(best_active_set)
        current_cc = compute_cover_count(current_set, deadline)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            active_list = list(current_set)
            inactive_list = [wi for wi in range(total_words) if wi not in current_set]
            
            if not active_list or not inactive_list:
                break
            
            random.shuffle(active_list)
            random.shuffle(inactive_list)
            
            for ai in active_list[:15]:
                if time.time() > deadline - 3.0:
                    break
                for bi in inactive_list[:15]:
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Simulate swap: remove ai, add bi
                    # Check validity: after removal, all words still covered by adding bi
                    
                    # Words that become uncovered after removing ai
                    # (those with count 1 and covered by ai)
                    zero_words = []
                    for w in covers_all[ai]:
                        if current_cc[w] == 1:
                            zero_words.append(w)
                    
                    if zero_words:
                        # Check if bi covers all zero words
                        bi_covers = set(covers_all[bi])
                        valid = True
                        for w in zero_words:
                            if w not in bi_covers:
                                valid = False
                                break
                        if not valid:
                            continue
                    
                    # Perform swap
                    current_set.discard(ai)
                    for w in covers_all[ai]:
                        current_cc[w] -= 1
                    current_set.add(bi)
                    for w in covers_all[bi]:
                        current_cc[w] += 1
                    
                    # Try to remove more
                    greedy_removal_set(current_set, current_cc, deadline)
                    
                    if len(current_set) < best_count:
                        best_count = len(current_set)
                        best_active_set = set(current_set)
                        improved = True
                        break
                    else:
                        # Undo swap
                        current_set.discard(bi)
                        for w in covers_all[bi]:
                            current_cc[w] -= 1
                        current_set.add(ai)
                        for w in covers_all[ai]:
                            current_cc[w] += 1
                        # Undo any removals - just recompute
                        current_cc = compute_cover_count(current_set, deadline)
                
                if improved:
                    break
    
    # Phase 4: Final greedy removal
    final_cc = compute_cover_count(best_active_set, deadline)
    greedy_removal_set(best_active_set, final_cc, deadline)
    
    # Convert back to list of codeword lists
    result = []
    for wi in sorted(best_active_set):
        w = []
        tmp = wi
        for pos in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        result.append(w)
    
    return result
