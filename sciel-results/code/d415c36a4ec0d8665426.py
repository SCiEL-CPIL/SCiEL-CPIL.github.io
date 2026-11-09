"""Rewritten program: Critical Word Swap + Greedy Removal for ternary covering code K_3(8,1)."""
import time
import random
import numpy as np

# Hard wall-clock budget (seconds). The evaluator kills at 150s; keep a safety margin.
DEADLINE_SECONDS = 140.0


def construct(q, n, radius):
    start_time = time.time()
    
    # Precompute powers of q for index conversion
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i - 1] * q
    
    total_words = q ** n
    
    def idx_to_digits(idx):
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * powers[n - 1 - i]
        return idx

    # Generate all neighbors for every word in the space.
    # neighbor_list[i] = list of neighbor indices of word i (including i)
    neighbor_list = [[] for _ in range(total_words)]
    for i in range(total_words):
        digits = idx_to_digits(i)
        neighbor_list[i].append(i)
        for pos in range(n):
            for new_val in range(q):
                if new_val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                j = digits_to_idx(new_digits)
                neighbor_list[i].append(j)
        
    # Initial solution: Direct sum construction (Tetracode x {0,1,2}^4)
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    codeword_set = set()
    num_tails = q ** (n - 4)
    for h in tetracode:
        for tail_idx in range(num_tails):
            tail_digits = [0] * (n - 4)
            temp = tail_idx
            for i in range(n - 5, -1, -1):
                tail_digits[i] = temp % q
                temp //= q
            full_word = h + tail_digits
            idx = digits_to_idx(full_word)
            codeword_set.add(idx)
    
    codewords = list(codeword_set)
    
    # Coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for nb in neighbor_list[cw]:
            coverage_count[nb] += 1
            
    # Helper to check if all covered
    def is_valid():
        return np.all(coverage_count > 0)
        
    # Main loop: Greedy Removal and Critical Word Swaps
    # We will iterate until no more improvements or time runs out.
    
    while time.time() < DEADLINE_SECONDS - 5:
        # Phase 1: Greedy Removal of non-essential codewords
        # A codeword is removable if it covers no word with coverage_count == 1
        removable_candidates = []
        for cw in codewords:
            removable = True
            for nb in neighbor_list[cw]:
                if coverage_count[nb] == 1:
                    removable = False
                    break
            if removable:
                removable_candidates.append(cw)
                
        if removable_candidates:
            # Remove one random removable codeword
            cw_to_remove = random.choice(removable_candidates)
            for nb in neighbor_list[cw_to_remove]:
                coverage_count[nb] -= 1
            codewords.remove(cw_to_remove)
            continue # Restart the loop to find more removals
        
        # Phase 2: Critical Word Swap
        # Find words with coverage_count == 1
        critical_words = np.where(coverage_count == 1)[0]
        
        if len(critical_words) == 0:
            # Should not happen if we removed all removable codewords, 
            # unless the code is minimal and no swaps help.
            # If we are stuck, we might need a more aggressive move.
            # For now, break to avoid infinite loop if no progress.
            # But maybe we can try a random swap that maintains validity?
            # Let's try to find a swap that reduces "max coverage" or something?
            # Or just break.
            break
            
        # Pick a random critical word
        w = random.choice(critical_words)
        
        # Find the codeword c that covers w
        # Since coverage_count[w] == 1, there is exactly one such codeword.
        # We can find it by checking which codeword has w in its neighbor list.
        # Optimization: Precompute for each word, the list of codewords covering it?
        # That's expensive. Instead, iterate through codewords? O(N * 17).
        # Or, since we have neighbor_list, we can check if w is in neighbor_list[cw].
        # Actually, we can just find the unique codeword.
        c = None
        for cw in codewords:
            if w in neighbor_list[cw]:
                c = cw
                break
        
        if c is None:
            # Should not happen
            break
            
        # Find alternative codewords c' that cover w
        # c' must be in neighbor_list[w]
        # c' must not be in codeword_set
        # c' must be such that replacing c with c' keeps all words covered
        
        candidates = []
        for cw_prime in neighbor_list[w]:
            if cw_prime in codeword_set:
                continue
            # Check if replacing c with cw_prime is valid
            # Words covered by c but not cw_prime must be covered by other codewords.
            # i.e., for all nb in neighbor_list[c], if nb not in neighbor_list[cw_prime], 
            # then coverage_count[nb] >= 2 (since c is the only one covering it? No, 
            # coverage_count[nb] includes c. So we need coverage_count[nb] - 1 >= 1 
            # if nb is not covered by cw_prime).
            # More simply: 
            # New coverage = coverage_count - neighbor_list[c] + neighbor_list[cw_prime]
            # We need new_coverage > 0 everywhere.
            
            # Let's check the "loss" from removing c
            # Loss set: {nb in neighbor_list[c] such that nb not in neighbor_list[cw_prime]}
            # For each nb in Loss set, we need coverage_count[nb] > 1.
            
            valid_swap = True
            # Optimization: Only check neighbors of c
            for nb in neighbor_list[c]:
                if nb not in neighbor_list[cw_prime]:
                    if coverage_count[nb] == 1:
                        valid_swap = False
                        break
            
            if valid_swap:
                candidates.append(cw_prime)
                
        if candidates:
            # Pick a random candidate
            cw_prime = random.choice(candidates)
            
            # Perform swap
            for nb in neighbor_list[c]:
                coverage_count[nb] -= 1
            for nb in neighbor_list[cw_prime]:
                coverage_count[nb] += 1
            
            codewords.remove(c)
            codewords.append(cw_prime)
            
            # Continue loop to try removals again
            continue
        else:
            # No valid swap found for this critical word.
            # Try another critical word? Or break?
            # To avoid getting stuck, we can try a random swap that doesn't necessarily fix a critical word
            # but might change the structure.
            # For now, let's just break if we can't find a swap for a random critical word.
            # A better strategy: Shuffle critical words and try a few.
            # But for simplicity, break.
            break

    # Final Greedy Removal pass
    changed = True
    while changed and time.time() < DEADLINE_SECONDS - 2:
        changed = False
        removable_candidates = []
        for cw in codewords:
            removable = True
            for nb in neighbor_list[cw]:
                if coverage_count[nb] == 1:
                    removable = False
                    break
            if removable:
                removable_candidates.append(cw)
        if removable_candidates:
            cw_to_remove = random.choice(removable_candidates)
            for nb in neighbor_list[cw_to_remove]:
                coverage_count[nb] -= 1
            codewords.remove(cw_to_remove)
            changed = True

    # Convert to list of lists
    result = []
    for idx in codewords:
        result.append(idx_to_digits(idx))
    
    return result

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct(3, 8, 1)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Codewords: {len(res)}")
    pass
