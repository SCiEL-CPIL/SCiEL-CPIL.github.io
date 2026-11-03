import time

def construct(n):
    """
    Constructs a superpermutation of length 1! + 2! + ... + n!.
    This is the classic recursive construction.
    """
    start_time = time.time()
    # Allow 140s for computation, 10s buffer
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        perms = []
        seen = set()
        
        # Extract permutations of length m-1
        # Optimization: only check windows that could be permutations
        # Since m is small (up to 7), simple tuple conversion is fine.
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Quick check for duplicates
            if len(set(window)) < m - 1:
                continue
            w_tuple = tuple(window)
            if w_tuple not in seen:
                seen.add(w_tuple)
                perms.append(w_tuple)
                
        new_word = []
        for p in perms:
            # Block: p, m, p
            # Find overlap with current end of new_word
            # Max possible overlap is min(len(new_word), len(p) + 1 + len(p))
            # But realistically, overlap is with the start of the block.
            # Block starts with p.
            # So we check if suffix of new_word matches prefix of p.
            max_overlap = min(len(new_word), len(p))
            overlap = 0
            for o in range(max_overlap, 0, -1):
                if new_word[-o:] == list(p[:o]):
                    overlap = o
                    break
            
            if overlap == 0:
                new_word.extend(list(p))
                new_word.append(m)
                new_word.extend(list(p))
            else:
                # Append the rest of the block
                # Block is p + [m] + p
                # We already have p[:overlap] at the end of new_word?
                # No, we matched new_word suffix with p prefix.
                # So we need to append p[overlap:] + [m] + p
                new_word.extend(list(p[overlap:]))
                new_word.append(m)
                new_word.extend(list(p))
                
        word = new_word
        
        if time.time() > deadline:
            # If we run out of time, we return the current partial word?
            # No, that would be invalid.
            # But the classic construction is very fast for n=7.
            # This guard is just for safety against unexpected slowness.
            pass

    return "".join(map(str, word))
