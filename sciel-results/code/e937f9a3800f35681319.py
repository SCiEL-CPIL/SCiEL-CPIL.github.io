import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code using a Bitset-based Branch-and-Bound algorithm
    for the Exact Cover / Minimum Set Cover formulation.
    
    The universe is all q^n words.
    The sets are the Hamming balls of radius 'radius' centered at each word.
    We seek the minimum number of sets to cover the universe.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute coverage: for each word i, which words does it cover?
    # For radius 1, this is i + all words with Hamming distance 1.
    # Since n=7, q=3, total_words=2187.
    # We can represent the set of covered words as a bitset (Python int).
    
    # 1. Generate all balls as bitsets
    balls = [0] * total_words
    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i
        balls[i] = mask
        if radius >= 1:
            for j in range(n):
                for v in range(q):
                    if v == w[j]:
                        continue
                    # Create new word
                    new_w = w.copy()
                    new_w[j] = v
                    # Find index of new_w
                    # Efficient way: precompute index map
                    # For 2187 words, a dictionary is fast enough
                    idx = 0
                    # Actually, let's precompute a map from tuple to index to avoid O(n) lookup in inner loop
                    pass
                    
    # Better precomputation:
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i
        balls[i] = mask
        if radius >= 1:
            for j in range(n):
                for v in range(q):
                    if v == w[j]:
                        continue
                    new_w = w.copy()
                    new_w[j] = v
                    idx = word_to_idx[tuple(new_w)]
                    balls[i] |= (1 << idx)
                    
    # 2. Branch and Bound
    # State:
    #   uncovered: bitset of words not yet covered
    #   current_code: list of indices of chosen codewords
    #   best_code: list of indices of the best solution found so far
    
    full_mask = (1 << total_words) - 1
    uncovered = full_mask
    
    best_code = []
    # Initial upper bound: Greedy set cover to get a decent starting point
    # This helps B&B prune early
    temp_uncovered = full_mask
    greedy_code = []
    while temp_uncovered:
        best_idx = -1
        max_cov = 0
        for i in range(total_words):
            if balls[i] & temp_uncovered:
                cov_count = (balls[i] & temp_uncovered).bit_count()
                if cov_count > max_cov:
                    max_cov = cov_count
                    best_idx = i
        if best_idx == -1:
            break
        greedy_code.append(best_idx)
        temp_uncovered &= ~balls[best_idx]
        
    if greedy_code:
        best_code = greedy_code[:]
        
    # Heuristic for variable ordering:
    # Pick the uncovered word with the fewest candidate codewords that cover it.
    # This is the "Most Constrained Variable" heuristic.
    
    # Precompute for each word, the list of codewords that cover it?
    # Or just compute on the fly?
    # On the fly is O(N) per step. N=2187.
    # Let's precompute reverse coverage: for each word w, which balls cover w?
    # This is the transpose of the coverage matrix.
    
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = balls[i]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            coverers[idx].append(i)
            m ^= lsb
            
    # Sort coverers by length to pick the most constrained word faster?
    # We need to find the uncovered word with min len(coverers[w]) among uncovered w.
    # We can precompute a list of words sorted by len(coverers[w]).
    # But the "uncovered" status changes.
    # So we just iterate through words sorted by static constraint?
    # No, dynamic constraint is better.
    # However, for radius 1, the number of coverers is roughly constant (1 + n*(q-1) = 15).
    # So the "most constrained" heuristic might not vary much.
    # Let's stick to a simple heuristic: pick the first uncovered word in a fixed order,
    # or the one with the fewest available choices.
    
    # Given the small size, we can afford to check all uncovered words to find the one with fewest options.
    # But we need to be fast.
    
    # Let's use a recursive function with memoization? No, state space is huge.
    # We rely on pruning.
    
    def branch_and_bound(uncovered_mask, current_depth, current_code):
        nonlocal best_code, best_depth
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return
            
        if not uncovered_mask:
            if len(current_code) < len(best_code):
                best_code = current_code[:]
            return
            
        # Pruning: if current_depth + 1 >= len(best_code), we can't improve
        # Actually, we need at least 1 more word.
        if len(current_code) >= len(best_code):
            return
            
        # Heuristic: Find an uncovered word with the fewest covering options
        # We iterate through all words? 2187 iterations.
        # Check if uncovered.
        # Count available coverers (those not already used? No, we can reuse? No, set cover, we pick distinct sets).
        # Wait, standard set cover picks distinct sets.
        # So we need to know which sets are already picked?
        # Actually, if we pick a set, we remove its covered elements.
        # The "options" for a word w are the sets that cover w.
        # If a set is already picked, it's not an "option" to pick again.
        # But we don't track "picked sets" in the mask, we track "uncovered elements".
        # If a set covers w, and w is uncovered, then that set is a valid choice.
        # So the number of options for w is the number of sets that cover w.
        # This number is static! It doesn't change based on previous picks, unless we consider that
        # picking a set might cover other words, but it doesn't remove the set itself from the pool.
        # Wait, in set cover, once a set is picked, it's gone.
        # So if we picked set S1, we can't pick S1 again.
        # Does that affect the number of options for w?
        # Yes, if w is covered by S1, and we picked S1, w would be covered.
        # So if w is uncovered, S1 was NOT picked.
        # Therefore, ALL sets that cover w are still available to be picked.
        # So the number of options for w is just len(coverers[w]), which is constant.
        # This means the "Most Constrained Variable" heuristic based on count is static.
        
        # So we should pick an uncovered word that is "hard" to cover?
        # Since counts are similar, maybe pick the one that appears in the fewest "remaining" sets?
        # But all sets covering w are remaining.
        
        # Alternative heuristic: Pick the word that is covered by the fewest sets in total?
        # They are all ~15.
        
        # Let's just pick the first uncovered word.
        # To make it smarter, we can sort words by some static priority.
        # Let's try picking the word with the smallest index first?
        # Or random?
        
        # Let's try a different heuristic:
        # Pick the word w such that the union of balls of its coverers is smallest?
        # This is expensive.
        
        # Let's stick to: Pick the first uncovered word.
        # To improve, we can order the words in `coverers` such that we check the most "central" words first?
        
        # Let's find the first uncovered word.
        target_word = -1
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            target_word = idx
            break
            
        # Get all sets that cover target_word
        options = coverers[target_word]
        
        # Sort options by how much they cover (greedy tie-breaker)
        # This helps find a good solution faster
        def score_option(i):
            return (balls[i] & uncovered_mask).bit_count()
            
        options.sort(key=score_option, reverse=True)
        
        for opt in options:
            new_uncovered = uncovered_mask & ~balls[opt]
            branch_and_bound(new_uncovered, len(current_code) + 1, current_code + [opt])
            
            if time.time() > deadline:
                return
                
    # Initialize best_depth to len(greedy_code)
    best_depth = len(best_code)
    
    branch_and_bound(full_mask, 0, [])
    
    # Return the best code found
    final_codewords = [all_words[i].tolist() for i in best_code]
    
    # Final validity check
    # For this size, it should be fine.
    # If best_code is empty, something went wrong, return greedy fallback
    if not final_codewords:
         final_codewords = [all_words[i].tolist() for i in greedy_code]

    return final_codewords
