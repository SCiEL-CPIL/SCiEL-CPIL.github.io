"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    """
    Constructs a superpermutation for n symbols using the standard recursive method.
    Length is sum_{i=1}^n i!.
    For n=7, length is 5913.
    """
    if n <= 0:
        return ""
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all unique permutations of length m-1 from the current word
        # in the order of their first appearance.
        perms = []
        seen = set()
        
        # The window size is m-1
        window_size = m - 1
        for i in range(len(word) - window_size + 1):
            window = tuple(word[i:i + window_size])
            # Check if it's a valid permutation (all distinct)
            if len(set(window)) == window_size and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            # We only need to check up to min(len(new_word), len(block))
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
