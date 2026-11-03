"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    total_space = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w
    
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks
    cover_masks = []
    for w in all_words:
        mask = 1 << word_to_idx(w)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    mask |= (1 << word_to_idx(nw))
        cover_masks.append(mask)
    
    # Initial code from tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    code_idx = [word_to_idx(cw) for cw in initial]
    
    def greedy_reduce(idx_list):
        if len(idx_list) <= 1:
            return idx_list[:]
        masks = [cover_masks[i] for i in idx_list]
        full_union = 0
        for m in masks:
            full_union |= m
        n_cw = len(idx_list)
        keep = [True] * n_cw
        for i in range(n_cw):
            if time.time() > deadline:
                break
            if not keep[i]:
                continue
            union_without = 0
            for j in range(n_cw):
                if j != i and keep[j]:
                    union_without |= masks[j]
            if union_without == full_union:
                keep[i] = False
        return [idx_list[i] for i in range(n_cw) if keep[i]]
    
    def local_search(idx_list, max_rounds=30):
        code = idx_list[:]
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            n_cw = len(code)
            masks = [cover_masks[i] for i in code]
            full_union = 0
            for m in masks:
                full_union |= m
            for i in range(n_cw):
                if time.time() > deadline:
                    break
                union_without = 0
                for j in range(n_cw):
                    if j != i:
                        union_without |= masks[j]
                if union_without == full_union:
                    code.pop(i)
                    masks.pop(i)
                    full_union = union_without
                    improved = True
                    break
            if improved:
                continue
            code_set = set(code)
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                union_without = 0
                for j in range(len(code)):
                    if j != i:
                        union_without |= masks[j]
                uncovered = full_union & ~union_without
                if uncovered == 0:
                    code.pop(i)
                    masks.pop(i)
                    improved = True
                    break
                best_gain = 0
                best_new_idx = -1
                best_new_mask = 0
                ci = code[i]
                w = all_words[ci]
                candidates = {ci}
                for pos in range(n):
                    for val in range(q):
                        if val != w[pos]:
                            nw = w[:]
                            nw[pos] = val
                            candidates.add(word_to_idx(nw))
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        for v1 in range(q):
                            if v1 == w[p1]:
                                continue
                            for v2 in range(q):
                                if v2 == w[p2]:
                                    continue
                                nw = w[:]
                                nw[p1] = v1
                                nw[p2] = v2
                                candidates.add(word_to_idx(nw))
                for cand in candidates:
                    if cand in code_set:
                        continue
                    m = cover_masks[cand]
                    new_union = union_without | m
                    if new_union != full_union:
                        continue
                    gain = bin(uncovered & m).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_new_idx = cand
                        best_new_mask = m
                if best_new_idx >= 0:
                    code_set.discard(code[i])
                    code[i] = best_new_idx
                    masks[i] = best_new_mask
                    code_set.add(best_new_idx)
                    improved = True
                    break
            if not improved:
                break
        return code
    
    code_idx = greedy_reduce(code_idx)
    code_idx = local_search(code_idx)
    code_idx = greedy_reduce(code_idx)
    
    return [all_words[i] for i in code_idx]
# EVOLVE-BLOCK-END
