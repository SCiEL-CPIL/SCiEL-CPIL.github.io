import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage
    coverage = [set() for _ in range(num_words)]
    
    if radius == 1:
        for i, w in enumerate(all_words):
            coverage[i].add(i)
            w_list = list(w)
            for pos in range(n):
                orig = w_list[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w_list[pos] = new_val
                    neighbor = tuple(w_list)
                    coverage[i].add(word_to_idx[neighbor])
                    w_list[pos] = orig
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            w_list = list(w)
            # Generate all words within distance radius
            # This is expensive for large n, but n=8 is small
            # Using recursion or iterative deepening
            def gen_neighbors(pos, dist):
                if dist == 0:
                    yield tuple(w_list)
                    return
                for p in range(pos, n):
                    orig = w_list[p]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        w_list[p] = new_val
                        yield from gen_neighbors(p+1, dist-1)
                    w_list[p] = orig
            
            for nb in gen_neighbors(0, radius):
                coverage[i].add(word_to_idx[nb])
                
    # Greedy Set Cover
    uncovered = set(range(num_words))
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        max_covered = 0
        
        for i in range(num_words):
            if i in code_indices:
                continue
            covered_count = len(uncovered & coverage[i])
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
                
        if best_idx == -1 or max_covered == 0:
            break
            
        code_indices.append(best_idx)
        uncovered -= coverage[best_idx]
        
    # Pruning
    current_code = list(code_indices)
    redundant_found = True
    while redundant_found:
        redundant_found = False
        if time.time() > deadline:
            break
            
        counts = [0] * num_words
        for c in current_code:
            for w_idx in coverage[c]:
                counts[w_idx] += 1
                
        for c in current_code:
            is_redundant = True
            for w_idx in coverage[c]:
                if counts[w_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                current_code.remove(c)
                redundant_found = True
                break
                
    final_code = [all_words[i] for i in current_code]
    return final_code
