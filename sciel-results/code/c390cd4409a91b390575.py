import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"
    
    # For n >= 5, we use the standard recursive construction.
    # This is the best known general method and produces a valid superpermutation.
    # The length is n! + (n-1)! + ... + 2! + 1 + n - 1.
    # For n=5: 120 + 24 + 6 + 2 + 1 + 4 = 157
    # For n=6: 720 + 120 + 24 + 6 + 2 + 1 + 5 = 878
    # For n=7: 5040 + 720 + 120 + 24 + 6 + 2 + 1 + 6 = 5919
    
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, we return the current valid word.
            # However, for n <= 7, the construction is very fast.
            # S_7 length is 5919. The loop runs 6 times.
            # The inner loop for m=7 iterates 720 times.
            # Each iteration does a string concat and overlap check.
            # Overlap check is O(m^2) worst case, but m is small.
            # Total operations are well within 110s.
            return word
        
        # Extract unique (m-1) permutations in order of appearance
        perms = []
        seen = set()
        # Optimization: Use a list for words if needed, but strings are fine for n<=7
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            
            # Find max overlap
            # Since m is small (max 7), and block length is 2*m-1 (max 13),
            # and new_word grows, we can limit the overlap check to len(block)-1
            # because we need at least 1 new char? No, we need to match prefix of block.
            # Max possible overlap is len(block)-1? 
            # If block is "12312" and new_word ends with "12312", overlap is 5.
            # But block starts with p. p is a permutation of 1..m-1.
            # The first char of block is p[0].
            # The last char of new_word is the last char of the previous block.
            # Previous block ended with p_prev.
            # So new_word ends with p_prev.
            # We are checking overlap with block = p + str(m) + p.
            # So we are checking if suffix of new_word matches prefix of p + str(m) + p.
            
            max_overlap = 0
            # The overlap cannot be larger than len(block) - 1?
            # Actually, if new_word ends with the entire block, that would mean
            # the block was already fully present?
            # But we are adding blocks in a specific order.
            # Let's just check up to min(len(new_word), len(block) - 1) to be safe?
            # Wait, if new_word ends with "12312" and block is "12312", overlap 5.
            # Then we append block[5:] which is "".
            # This is correct.
            # So limit is min(len(new_word), len(block)).
            
            limit = min(len(new_word), len(block))
            # Optimization: We know that the overlap must be such that the overlapping
            # part is a prefix of `block` and a suffix of `new_word`.
            # Also, the overlapping part cannot contain the new symbol `m` in the middle
            # unless it's at the very end of the overlap?
            # Actually, `block` is `p + str(m) + p`.
            # The symbol `m` appears only at index m-1 in `block`.
            # If the overlap includes this `m`, then `new_word` must also have `m` at that position.
            # Since `new_word` is built from previous blocks, it might contain `m`?
            # No, `new_word` is built from symbols 1..m-1 only?
            # Wait.
            # In the previous step (m-1), the word contained only 1..m-1.
            # So `new_word` at the start of the loop for `m` contains only 1..m-1.
            # `block` contains `m` at position m-1.
            # If the overlap length `o` is such that the `m` in `block` is within the overlap,
            # then `new_word` must have `m` at the corresponding position.
            # But `new_word` does NOT contain `m` yet!
            # Therefore, the overlap CANNOT include the position of `m` in `block`.
            # The position of `m` in `block` is `m-1` (0-indexed).
            # So the overlap length `o` must be <= m-1.
            # Because if `o > m-1`, then the overlap includes index `m-1` of `block`,
            # which is `m`. But `new_word` doesn't have `m`.
            
            # This is a huge optimization!
            # We only need to check overlaps up to m-1.
            
            limit = min(len(new_word), m - 1)
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
    return word
