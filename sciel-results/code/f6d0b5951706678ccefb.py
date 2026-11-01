import itertools
import time

def _standard_superperm(n):
    """
    Standard recursive construction of superpermutation.
    Length: 1! + 2! + ... + n!
    """
    if n <= 1:
        return "1"
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            # Find maximum overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def construct(n):
    if n <= 1:
        return "1"
    
    if n <= 6:
        # For n <= 6, the standard construction is optimal
        return _standard_superperm(n)
    
    if n == 7:
        start_time = time.time()
        # Set a deadline to ensure we return within 120s
        deadline = 110.0 
        
        # Precompute permutation ranks for fast lookup
        perms = list(itertools.permutations(range(1, n + 1)))
        perm_to_rank = {p: i for i, p in enumerate(perms)}
        total_perms = len(perms)
        target_mask = (1 << total_perms) - 1
        
        # Beam Search
        # State: (suffix_tuple, visited_mask, length, word_str)
        initial_perm = tuple(range(1, n + 1))
        initial_rank = perm_to_rank[initial_perm]
        initial_visited = 1 << initial_rank
        initial_word = "".join(map(str, initial_perm))
        
        beam = [(initial_perm, initial_visited, n, initial_word)]
        
        best_word = None
        best_len = n * total_perms # Upper bound (n * n!)
        
        # Standard length for comparison
        std_len = len(_standard_superperm(n))
        
        max_steps = std_len # We don't need to search beyond the standard length if we want to beat it
        
        for step in range(n, max_steps + 1):
            if time.time() > deadline:
                break
            
            new_beam = []
            for suffix, visited, length, word in beam:
                # If this state is a complete solution
                if visited == target_mask:
                    if length < best_len:
                        best_len = length
                        best_word = word
                    # Do not expand terminal states
                    continue
                
                for c in range(1, n + 1):
                    new_suffix = suffix[1:] + (c,)
                    rank = perm_to_rank.get(new_suffix, -1)
                    new_visited = visited
                    if rank != -1:
                        if not (visited >> rank) & 1:
                            new_visited = visited | (1 << rank)
                    
                    new_word = word + str(c)
                    new_beam.append((new_suffix, new_visited, length + 1, new_word))
            
            if not new_beam:
                break
                
            # Prune: Keep top 50 states by number of visited permutations
            # This greedy heuristic favors paths that discover new permutations quickly
            new_beam.sort(key=lambda x: -x[1].bit_count())
            
            beam = new_beam[:50]
            
            # If we already found a solution better than standard, we could stop early?
            # No, we want the shortest. But if we find 5905, and standard is 5913,
            # we might find 5904 later.
            # However, if time is running out, we should return the best found.
        
        if best_word is not None and best_len < std_len:
            return best_word
        else:
            return _standard_superperm(n)
    else:
        return _standard_superperm(n)
