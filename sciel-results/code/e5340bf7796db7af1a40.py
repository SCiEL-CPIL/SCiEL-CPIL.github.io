import time
import random
import math
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.monotonic()
    deadline = start_time + 115.0
    
    total = q ** n  # 729
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: covers[c] = set of word indices within distance 1 of c
    covers = [None] * total
    for i in range(total):
        w = all_words[i]
        cov = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                cov.add(word_to_idx[tuple(nw)])
        covers[i] = cov
    
    # Precompute covered_by: for each word, which codewords cover it
    covered_by = [set() for _ in range(total)]
    for c in range(total):
        for w in covers[c]:
            covered_by[w].add(c)
    
    # Precompute distance-2 neighbors for better swap candidates
    dist2 = [set() for _ in range(total)]
    for i in range(total):
        w1 = all_words[i]
        for j in range(i + 1, total):
            w2 = all_words[j]
            d = sum(1 for k in range(n) if w1[k] != w2[k])
            if d == 2:
                dist2[i].add(j)
                dist2[j].add(i)
    
    def greedy_init(seed):
        """Greedy set cover with random tie-breaking."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        
        while True:
            if time.monotonic() > deadline - 5:
                break
            uncovered_mask = (counts == 0).astype(np.int32)
            if uncovered_mask.sum() == 0:
                break
            
            best_gain = 0
            best_candidates = []
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for w in covers[c]:
                    if counts[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
            
            if best_gain <= 0 or not best_candidates:
                break
            
            best_c = r.choice(best_candidates)
            code.append(best_c)
            code_set.add(best_c)
            for w in covers[best_c]:
                counts[w] += 1
        
        return code, counts, code_set
    
    def polish(code, counts, code_set, time_budget):
        """Local search with 2-for-1 moves, swaps, and removals."""
        end_time = time.monotonic() + time_budget
        best_code = code[:]
        best_size = len(code)
        rng = random.Random(int(time.monotonic() * 1000) % 100000)
        
        def is_redundant(idx):
            for w in covers[idx]:
                if counts[w] == 1:
                    return False
            return True
        
        def remove_codeword(idx):
            code.remove(idx)
            code_set.discard(idx)
            for w in covers[idx]:
                counts[w] -= 1
        
        def add_codeword(idx):
            code.append(idx)
            code_set.add(idx)
            for w in covers[idx]:
                counts[w] += 1
        
        # Phase 1: Remove all redundant codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Phase 2: Simulated annealing with 2-for-1, swap, and add moves
        temp = 3.0
        cool_rate = 0.9997
        min_temp = 0.05
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.35:
                # Try 2-for-1: find c1, c2 in code such that some c3 covers union
                if len(code) >= 2:
                    c1, c2 = rng.sample(code, 2)
                    union_cov = covers[c1] | covers[c2]
                    if union_cov:
                        sample_word = rng.choice(list(union_cov))
                        candidate_c3s = covered_by[sample_word]
                        found_c3 = -1
                        for c3 in candidate_c3s:
                            if c3 in code_set:
                                continue
                            if union_cov.issubset(covers[c3]):
                                found_c3 = c3
                                break
                        if found_c3 != -1:
                            remove_codeword(c1)
                            remove_codeword(c2)
                            add_codeword(found_c3)
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        
            elif move_type < 0.75:
                # Try swap: replace one codeword with another
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    
                    if rng.random() < 0.5 and dist2[old_c]:
                        new_c = rng.choice(list(dist2[old_c]))
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    unique_old = 0
                    for w in covers[old_c]:
                        if counts[w] == 1 and w not in covers[new_c]:
                            unique_old += 1
                    
                    if unique_old == 0:
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for w in covers[old_c]:
                            counts[w] -= 1
                        for w in covers[new_c]:
                            counts[w] += 1
                    elif unique_old <= 2 and rng.random() < math.exp(-unique_old / temp):
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for w in covers[old_c]:
                            counts[w] -= 1
                        for w in covers[new_c]:
                            counts[w] += 1
                        
                        for w in range(total):
                            if counts[w] == 0:
                                for nc in covered_by[w]:
                                    if nc not in code_set:
                                        add_codeword(nc)
                                        break
            
            else:
                # Try remove (if redundant)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_redundant(c):
                        remove_codeword(c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size
    
    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    num_restarts = 20
    time_per_restart = (deadline - time.monotonic() - 5) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10:
            break
        
        seed = 42 + restart * 17
        code, counts, code_set = greedy_init(seed)
        
        if not code:
            continue
        
        polished_code, polished_size = polish(code, counts, code_set, time_per_restart - 0.5)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
        
        if overall_best_size <= 73:
            break
    
    if overall_best_code is None:
        code, counts, code_set = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result


def _generic_construct(q, n, radius):
    """Generic fallback for other parameters."""
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage
    covers = [None] * total
    for i in range(total):
        w = all_words[i]
        cov = {i}
        # Generate all words within Hamming distance 'radius'
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for new_vals in itertools.product(range(q), repeat=d):
                    nw = list(w)
                    for pos, val in zip(positions, new_vals):
                        nw[pos] = val
                    nw = tuple(nw)
                    if nw != w:
                        cov.add(word_to_idx[nw])
        covers[i] = cov
    
    # Greedy set cover
    counts = np.zeros(total, dtype=np.int32)
    code = []
    code_set = set()
    
    while True:
        uncovered = sum(1 for c in counts if c == 0)
        if uncovered == 0:
            break
        
        best_gain = 0
        best_c = -1
        for c in range(total):
            if c in code_set:
                continue
            gain = sum(1 for w in covers[c] if counts[w] == 0)
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_gain <= 0:
            break
        
        code.append(best_c)
        code_set.add(best_c)
        for w in covers[best_c]:
            counts[w] += 1
    
    result = []
    for idx in code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
