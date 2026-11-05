"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    if n <= 1:
        return '1' * n
    
    # Generate the list of permutations in the standard recursive order
    # L_1 = [(1,)]
    # L_n: for p in L_{n-1}, insert n at positions len(p) down to 0
    perms = [(1,)]
    
    for k in range(2, n + 1):
        new_perms = []
        for p in perms:
            # Insert k at position len(p) down to 0
            for i in range(len(p), -1, -1):
                new_p = p[:i] + (k,) + p[i:]
                new_perms.append(new_p)
        perms = new_perms
        
    # Construct the superpermutation string by greedy overlap
    if not perms:
        return ""
        
    word = list(perms[0])
    
    for i in range(1, len(perms)):
        prev = perms[i-1]
        curr = perms[i]
        
        # Find maximum overlap between suffix of prev and prefix of curr
        max_overlap = 0
        # Max possible overlap is n-1
        for k in range(n - 1, 0, -1):
            if prev[-k:] == curr[:k]:
                max_overlap = k
                break
                
        # Append the non-overlapping part of curr
        to_add = curr[max_overlap:]
        word.extend(to_add)
        
    return ''.join(map(str, word))
# EVOLVE-BLOCK-END
