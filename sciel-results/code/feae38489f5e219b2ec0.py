import time
import random
import itertools

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Precompute which words each codeword covers
    # For radius 1, a codeword covers itself and all words at distance 1
    # For general radius, we'd need more, but let's assume radius=1 for fallback
    if radius != 1:
        # Very basic fallback: just return all words
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    # Map word to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    # For each word, compute the set of indices it covers
    # This is expensive for large n, but fallback is for small cases
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    # Greedy: pick codeword that covers most uncovered
    uncovered_count = [len(c) for c in covers.values()]
    
    while not all(covered):
        # Find codeword with most uncovered neighbors
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    # Precompute neighbors for all 6561 words
    # Each word has 1 + n*(q-1) = 1 + 8*2 = 17 neighbors
    neighbors = [None] * total
    
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx
    
    # Build neighbor lists
    for idx in range(total):
        w = idx_to_word(idx)
        w_list = list(w)
        nbrs = [idx]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx(tuple(w_list)))
            w_list[pos] = orig
        neighbors[idx] = nbrs
    
    # Start with a good initial code
    # Use the tetracode product structure as initial
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            initial_code.append(tuple(h + list(tail)))
    
    code_set = set(initial_code)
    code_list = list(code_set)
    
    # Cover count array
    cover_count = [0] * total
    for c in code_list:
        c_idx = word_to_idx(c)
        for nbr in neighbors[c_idx]:
            cover_count[nbr] += 1
    
    start_time = time.time()
    deadline = start_time + 105  # Leave 15s margin
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    # Local search with removal and swap moves
    # We'll alternate between trying removals and swaps
    
    while time.time() < deadline:
        # Phase 1: Try to remove words
        removed_any = False
        # Shuffle for diversity
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            c = code_list[i]
            c_idx = word_to_idx(c)
            removable = True
            for nbr in neighbors[c_idx]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                for nbr in neighbors[c_idx]:
                    cover_count[nbr] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        if removed_any:
            continue
        
        # Phase 2: Try swaps to escape local minimum
        # Pick a random word, remove it, find a replacement that covers uncovered words
        if len(code_list) > 450:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            c_idx = word_to_idx(c)
            
            # Temporarily remove
            for nbr in neighbors[c_idx]:
                cover_count[nbr] -= 1
            
            # Find uncovered words
            uncovered = [nbr for nbr in neighbors[c_idx] if cover_count[nbr] == 0]
            
            if not uncovered:
                # Shouldn't happen, but if it does, keep the removal
                code_list.pop(idx_c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                continue
            
            # Find a replacement word that covers all uncovered words
            # Search among words not in code
            # For efficiency, only check words that are close to uncovered words
            # Actually, let's check all 6561 words but break early
            
            found_swap = False
            for idx_cp in range(total):
                if idx_cp == c_idx:
                    continue
                # Check if idx_cp covers all uncovered
                cp_nbrs_set = set(neighbors[idx_cp])
                all_covered = all(u in cp_nbrs_set for u in uncovered)
                if all_covered:
                    # Perform swap
                    code_list[idx_c] = idx_to_word(idx_cp)
                    for nbr in neighbors[idx_cp]:
                        cover_count[nbr] += 1
                    found_swap = True
                    break
            
            if not found_swap:
                # Revert
                for nbr in neighbors[c_idx]:
                    cover_count[nbr] += 1
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(c) for c in best_code]
    return [list(c) for c in code_list]
