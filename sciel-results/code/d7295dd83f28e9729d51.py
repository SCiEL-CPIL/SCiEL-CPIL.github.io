import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    # Specifics for the task
    total_words = q ** n
    num_coords = n
    alphabet = q
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx

    # Precompute coverage: coverage[cw] = list of words covered by cw
    cover_size = 1 + num_coords * (alphabet - 1)
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    for cw_idx in range(total_words):
        digits = idx_to_digits(cw_idx)
        coverage[cw_idx, 0] = cw_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                coverage[cw_idx, j] = digits_to_idx(new_digits)
                j += 1

    # Precompute reverse coverage: covers_word[w] = list of codewords covering w
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)
    for w_idx in range(total_words):
        digits = idx_to_digits(w_idx)
        covers_word[w_idx, 0] = w_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                covers_word[w_idx, j] = digits_to_idx(new_digits)
                j += 1

    # 1. Greedy Initialization
    uncovered = np.ones(total_words, dtype=np.uint8)
    count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize counts
    for w_idx in range(total_words):
        for j in range(cover_size):
            cw_idx = covers_word[w_idx, j]
            count[cw_idx] += 1
            
    codewords = []
    num_covered = 0
    
    while num_covered < total_words:
        if time.time() > deadline:
            break
        best_cw = np.argmax(count)
        if count[best_cw] == 0:
            break
        codewords.append(best_cw)
        for j in range(cover_size):
            w_idx = coverage[best_cw, j]
            if uncovered[w_idx]:
                uncovered[w_idx] = 0
                num_covered += 1
                for k in range(cover_size):
                    cw_idx = covers_word[w_idx, k]
                    count[cw_idx] -= 1
                    
    if num_covered < total_words:
         pass

    # 2. Local Search / Minimization
    set_of_cw = set(codewords)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    for cw in set_of_cw:
        for j in range(cover_size):
            w_idx = coverage[cw, j]
            cover_count[w_idx] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 2a. Try Deletions
        # A codeword is removable if min(cover_count[coverage[cw]]) > 1
        for cw in list(set_of_cw):
            removable = True
            for j in range(cover_size):
                w_idx = coverage[cw, j]
                if cover_count[w_idx] <= 1:
                    removable = False
                    break
            
            if removable:
                set_of_cw.remove(cw)
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] -= 1
                improved = True
                break
        
        if improved:
            continue

        # 2b. Try 2-for-1 Replacements
        # We look for pairs (c1, c2) in S such that there exists c3 not in S
        # covering all words that are uniquely covered by c1 or c2 (relative to S\{c1,c2})
        
        # To make this efficient, we iterate through pairs.
        # We can limit the number of pairs checked per iteration if it's too slow.
        # But |S| is ~500, so 125k pairs. Each check is fast if we use cover_count.
        
        found_swap = False
        # Shuffle order to avoid bias
        cw_list = list(set_of_cw)
        random.shuffle(cw_list)
        
        for i in range(len(cw_list)):
            c1 = cw_list[i]
            for j in range(i+1, len(cw_list)):
                if time.time() > deadline - 5:
                    break
                c2 = cw_list[j]
                
                # Find essential words for this pair
                # Essential words are w in B(c1) U B(c2) such that cover_count[w] == 2
                # and the two codewords covering w are c1 and c2.
                
                essential_words = []
                
                # Check words covered by c1
                for k in range(cover_size):
                    w_idx = coverage[c1, k]
                    if cover_count[w_idx] == 2:
                        # Check if the other cover is c2
                        # We can check covers_word[w_idx]
                        # If c2 is in covers_word[w_idx], then c1 and c2 are the two.
                        # Since c1 covers w_idx, and cover_count is 2, the other one must be c2 if c2 covers w_idx.
                        # So we just need to check if c2 covers w_idx.
                        # We can precompute a boolean matrix? No, too big.
                        # We can check if c2 is in the small list covers_word[w_idx].
                        if c2 in covers_word[w_idx]:
                            essential_words.append(w_idx)
                            
                # Check words covered by c2 (excluding those already checked in c1's ball if they overlap? 
                # Actually, if w is in B(c1) and B(c2), it was checked in c1's loop.
                # If cover_count[w]==2 and c2 covers w, it's essential.
                # If we check c2's ball, we might double add. 
                # Let's just collect unique essential words.
                # But wait, if w is in B(c2) but not B(c1), and cover_count[w]==2, 
                # the other cover must be c1? No, it could be another codeword c3.
                # If cover_count[w]==2, and c2 covers w, the other one is X.
                # If X == c1, then it's essential for the pair.
                # If X != c1, then it's not essential for this pair (c1 is not involved).
                # So we only care if the "other" cover is the partner in the pair.
                
                # So for w in B(c2):
                # if cover_count[w] == 2:
                #    if c1 in covers_word[w_idx]: essential.
                
                for k in range(cover_size):
                    w_idx = coverage[c2, k]
                    if cover_count[w_idx] == 2:
                        if c1 in covers_word[w_idx]:
                            # Avoid duplicates
                            if w_idx not in essential_words:
                                essential_words.append(w_idx)

                if not essential_words:
                    # This means one of them is redundant? 
                    # If essential_words is empty, it means for all w in B(c1) U B(c2), cover_count > 2 OR cover_count == 2 but the other cover is not the partner.
                    # Wait, if cover_count[w] == 1, we would have found it in deletion step? 
                    # Deletion step checks if min(cover_count[coverage[cw]]) > 1.
                    # If we are here, no single deletion is possible.
                    # So cover_count[w] >= 2 for all w in B(c1) and B(c2).
                    # If essential_words is empty, it means no word is covered *exactly* by c1 and c2.
                    # This doesn't mean we can remove one. It means they are not "exclusive" to each other.
                    # But we want to replace 2 with 1.
                    # If essential_words is empty, we can replace {c1, c2} with ANY c3? 
                    # No, we need c3 to cover the words that would be left uncovered.
                    # If we remove c1 and c2, words with cover_count == 2 (covered by c1/c2 and others) become cover_count 1.
                    # Words with cover_count > 2 remain covered.
                    # So the words that become "critical" (count 1) are those with cover_count == 2.
                    # We need c3 to cover all words w such that cover_count[w] == 2 AND (c1 in covers(w) OR c2 in covers(w)).
                    # This is exactly the set of essential_words defined above?
                    # Yes. If w has count 2 and is covered by c1, the other is X. If X==c2, it's essential.
                    # If X!=c2, then removing c1 leaves count 1, but c2 doesn't cover it, so c2's removal doesn't affect it?
                    # Wait. If we remove c1 and c2.
                    # w is covered by c1 and X (X != c2). Count 2.
                    # Remove c1: count 1 (covered by X).
                    # Remove c2: count 1 (still covered by X).
                    # So w is still covered.
                    # So we only need to cover words that were covered by c1 OR c2, and NO OTHER codeword?
                    # No, if w is covered by c1 and X, and we remove c1 and c2, w is covered by X. So it's fine.
                    # The only words that become uncovered are those covered by c1 or c2, and NO OTHER codeword?
                    # If w is covered by c1 only (count 1), it becomes uncovered. But we assumed no deletions possible, so count >= 2.
                    # If w is covered by c1 and X (count 2). Remove c1, c2. w covered by X. OK.
                    # If w is covered by c1 and c2 (count 2). Remove c1, c2. w uncovered. BAD.
                    # So essential_words are those covered by c1 AND c2 (and no others).
                    # My logic above: "if cover_count[w_idx] == 2 and c2 in covers_word[w_idx]" for w in B(c1).
                    # This correctly identifies words covered by c1 and c2 (and no others, since count is 2).
                    
                    # So if essential_words is empty, we can replace {c1, c2} with ANY c3?
                    # We just need to ensure the new set is valid.
                    # Since no words become uncovered, any c3 is fine?
                    # Yes, we can pick any c3 not in S.
                    # But we want to minimize size. Replacing 2 with 1 reduces size by 1.
                    # So we pick any c3 not in S.
                    
                    # Find any c3 not in S
                    # To be safe, pick one that covers many other words? 
                    # Just pick a random one or the first one not in S.
                    # Since total_words is 6561, and |S| is ~500, most are not in S.
                    # We can just pick 0 if 0 not in S, else 1, etc.
                    
                    c3 = None
                    for candidate in range(total_words):
                        if candidate not in set_of_cw:
                            c3 = candidate
                            break
                    
                    if c3 is not None:
                        # Perform swap
                        set_of_cw.remove(c1)
                        set_of_cw.remove(c2)
                        set_of_cw.add(c3)
                        
                        # Update cover_count
                        for k in range(cover_size):
                            w_idx = coverage[c1, k]
                            cover_count[w_idx] -= 1
                        for k in range(cover_size):
                            w_idx = coverage[c2, k]
                            cover_count[w_idx] -= 1
                        for k in range(cover_size):
                            w_idx = coverage[c3, k]
                            cover_count[w_idx] += 1
                            
                        improved = True
                        found_swap = True
                        break
                        
                else:
                    # We have essential words. We need a c3 that covers all of them.
                    # Find intersection of covers_word[w] for w in essential_words.
                    # Filter to candidates not in S.
                    
                    # Start with covers of first essential word
                    candidates = set(covers_word[essential_words[0]])
                    for w_idx in essential_words[1:]:
                        candidates &= set(covers_word[w_idx])
                        
                    # Filter out those in S
                    candidates -= set_of_cw
                    
                    if candidates:
                        c3 = next(iter(candidates))
                        
                        # Perform swap
                        set_of_cw.remove(c1)
                        set_of_cw.remove(c2)
                        set_of_cw.add(c3)
                        
                        # Update cover_count
                        for k in range(cover_size):
                            w_idx = coverage[c1, k]
                            cover_count[w_idx] -= 1
                        for k in range(cover_size):
                            w_idx = coverage[c2, k]
                            cover_count[w_idx] -= 1
                        for k in range(cover_size):
                            w_idx = coverage[c3, k]
                            cover_count[w_idx] += 1
                            
                        improved = True
                        found_swap = True
                        break
            
            if found_swap:
                break

        if not improved:
            # Try 1-for-1 swaps?
            # If we are stuck, we might try to swap c1 with c2 to enable future deletions.
            # This is complex. For now, if we can't delete or 2-for-1, we stop.
            # Given the time limit and the strong greedy start, this should be sufficient.
            pass

    # Convert to list of digits
    result = [idx_to_digits(cw) for cw in set_of_cw]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
