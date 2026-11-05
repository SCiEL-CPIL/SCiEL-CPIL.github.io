"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    if n <= 1:
        return '1' * n
    
    # Use naive construction: concatenate all permutations
    # This guarantees a valid superpermutation
    # Length is n * n!
    symbols = list(range(1, n + 1))
    perms = itertools.permutations(symbols)
    
    word_chars = []
    for p in perms:
        word_chars.extend(p)
        
    return ''.join(map(str, word_chars))
# EVOLVE-BLOCK-END
