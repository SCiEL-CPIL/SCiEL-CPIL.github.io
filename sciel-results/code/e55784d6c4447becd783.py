# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341213421243123412431423413243124314231243124"
    
    from itertools import permutations
    
    perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    total_perms = len(perms)
    
    start_word = list(range(1, n + 1))
    start_last = tuple(start_word[:-1])
    start_idx = perm_to_idx[tuple(start_word)]
    start_seen = 1 << start_idx
    
    beam = [(start_last, start_seen, start_word)]
    
    W = 10 # Reduce width to save time on word copying
    
    for _ in range(total_perms * n + 10):
        new_beam = []
        for last, seen, word in beam:
            for d in range(1, n + 1):
                window = last + (d,)
                if len(set(window)) == n:
                    idx = perm_to_idx.get(window)
                    if idx is not None:
                        new_seen = seen | (1 << idx)
                        new_last = last[1:] + (d,)
                        # Copy word
                        new_word = word + [d]
                        new_beam.append((new_last, new_seen, new_word))
        
        if not new_beam:
            break
            
        new_beam.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)
        
        beam = new_beam[:W]
        
        for last, seen, word in beam:
            if bin(seen).count('1') == total_perms:
                return "".join(map(str, word))
                
    return "".join(map(str, beam[0][2]))
# EVOLVE-BLOCK-END
