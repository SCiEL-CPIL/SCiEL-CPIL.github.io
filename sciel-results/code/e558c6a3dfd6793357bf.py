"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols using the standard
    recursive construction of length sum(k! for k in 1..n).
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        # The window size is m-1
        # We slide a window of size m-1 over the current word
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all distinct)
            # Since we build from valid perms, this should always be true, 
            # but checking len(set) is safe.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        # Process each permutation
        for p in perms:
            # Block is p, then m, then p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between new_word and block
            # We want the largest k such that new_word[-k:] == block[:k]
            max_overlap = 0
            # The overlap cannot exceed len(new_word) or len(block)
            # Also, overlap must be at least 0.
            # We check from largest possible overlap down to 0.
            # Optimization: The overlap is typically small, but can be up to m-1?
            # Actually, the end of the previous block is ...p_prev.
            # The start of the current block is p_curr.
            # p_prev and p_curr are permutations of 1..m-1.
            # The suffix of the previous block is the permutation p_prev.
            # The prefix of the current block is the permutation p_curr.
            # They can only overlap if p_prev is a suffix of p_curr? No.
            # The previous block ends with p_prev.
            # The current block starts with p_curr.
            # We need suffix of new_word to match prefix of block.
            # new_word ends with p_prev.
            # So we are looking for overlap between p_prev (suffix) and p_curr (prefix).
            # Since p_prev and p_curr are permutations of the same set, 
            # the only way a suffix of p_prev matches a prefix of p_curr is if they are identical?
            # No, e.g., p_prev=123, p_curr=231.
            # Suffixes of 123: 3, 23, 123.
            # Prefixes of 231: 2, 23, 231.
            # Match: 23. Overlap 2.
            
            # Efficient overlap check
            # The maximum possible overlap is m-1 (the length of the permutation part).
            # Actually, can it be larger?
            # The block is p + [m] + p.
            # The previous block ended with p_prev.
            # So new_word ends with p_prev.
            # The current block starts with p_curr.
            # We check overlap of p_prev and p_curr.
            # Let k be the overlap.
            # p_prev[-k:] == p_curr[:k]
            
            # We iterate k from min(len(new_word), len(block)) down to 0?
            # No, we only need to check up to m-1 because the 'm' in the middle
            # of the block prevents overlap extending into the 'm' part of the previous block
            # unless the previous block also had an 'm' at that position, which it didn't 
            # (it ended with p_prev).
            # So max overlap is m-1.
            
            # However, new_word might have more than just p_prev at the end?
            # No, the previous block was p_prev + [m-1] + p_prev.
            # So new_word ends with p_prev.
            # So we just need to find max k <= m-1 such that 
            # new_word[-k:] == block[:k].
            
            # Let's just do a linear scan from m-1 down to 0.
            # m is small (max 7).
            
            max_possible_overlap = min(len(new_word), len(block), m - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if new_word[len(new_word)-k:] == block[:k]:
                    overlap = k
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            # If we run out of time, we return what we have so far.
            # This shouldn't happen for n=7 with this algorithm.
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
