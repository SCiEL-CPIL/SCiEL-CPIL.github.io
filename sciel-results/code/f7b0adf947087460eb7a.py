import random
import time

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _construct_fallback(q, n, radius)

    total_words = q ** n
    indices = range(total_words)
    
    def tuple_to_index(t):
        idx = 0
        for digit in t:
            idx = idx * q + digit
        return idx

    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute ball masks
    ball_masks = [0] * total_words
    for i in indices:
        w = index_to_tuple(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = tuple_to_index(tuple(new_w))
                    mask |= (1 << j)
        ball_masks[i] = mask

    all_bits = (1 << total_words) - 1

    def local_search(codewords):
        improved = True
        while improved:
            improved = False
            # 1. Remove redundant
            for i in range(len(codewords)):
                union_others = 0
                for j in range(len(codewords)):
                    if j != i:
                        union_others |= ball_masks[codewords[j]]
                if (union_others & all_bits) == all_bits:
                    codewords.pop(i)
                    improved = True
                    break
            
            if len(codewords) == 0:
                break
                
            # 2. Try swaps
            if not improved:
                for _ in range(100):
                    i = random.randint(0, len(codewords) - 1)
                    cand = random.randint(0, total_words - 1)
                    if cand in codewords:
                        continue
                    union_new = ball_masks[cand]
                    for j in range(len(codewords)):
                        if j != i:
                            union_new |= ball_masks[codewords[j]]
                    if (union_new & all_bits) == all_bits:
                        codewords[i] = cand
                        improved = True
                        break
        return codewords

    def greedy_fill(uncovered_mask, existing_codewords):
        codewords = list(existing_codewords)
        uncovered = uncovered_mask
        while uncovered:
            max_cov = -1
            best_idx = -1
            for i in indices:
                if i in codewords:
                    continue
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return codewords

    def beam_search(beam_width=100, max_depth=100):
        beam = [(0, ())]
        
        for depth in range(max_depth):
            next_beam_candidates = []
            for mask, cwords in beam:
                if mask == all_bits:
                    return list(cwords)
                uncovered = all_bits & ~mask
                for i in indices:
                    new_mask = mask | ball_masks[i]
                    if new_mask == mask:
                        continue
                    next_beam_candidates.append((new_mask, cwords + (i,)))
            
            if not next_beam_candidates:
                break
            
            next_beam_candidates.sort(key=lambda x: (-(x[0].bit_count()), len(x[1])))
            
            seen_masks = {}
            pruned_beam = []
            for mask, cwords in next_beam_candidates:
                if mask in seen_masks:
                    continue
                else:
                    seen_masks[mask] = cwords
                    pruned_beam.append((mask, cwords))
                
                if len(pruned_beam) >= beam_width:
                    break
            
            beam = pruned_beam
            
            if not beam:
                break
                
        if beam:
            best = max(beam, key=lambda x: x[0].bit_count())
            mask, cwords = best
            uncovered = all_bits & ~mask
            return greedy_fill(uncovered, cwords)
        return []

    start_time = time.time()
    time_limit = 115

    best_codewords = beam_search(beam_width=150)
    best_size = len(best_codewords)
    
    best_codewords = local_search(best_codewords)
    best_size = len(best_codewords)

    current = best_codewords[:]
    
    while time.time() - start_time < time_limit:
        current = local_search(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_codewords = current[:]
            
        if best_size == 73:
            break
            
        if len(current) > 0:
            i = random.randint(0, len(current) - 1)
            cand = random.randint(0, total_words - 1)
            if cand not in current:
                union_new = ball_masks[cand]
                for j in range(len(current)):
                    if j != i:
                        union_new |= ball_masks[current[j]]
                if (union_new & all_bits) == all_bits:
                    current[i] = cand
                    current = local_search(current)
                    if len(current) < best_size:
                        best_size = len(current)
                        best_codewords = current[:]

    result = [list(index_to_tuple(i)) for i in best_codewords]
    return result

def _construct_fallback(q, n, radius):
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1:
            break
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
    return [list(c) for c in codewords]
