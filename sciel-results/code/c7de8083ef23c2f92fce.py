import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    if radius != 1:
        return [list(w) for w in all_words]
    covered = [False] * total
    codewords = []
    word_to_idx = {w: i for i, w in enumerate(all_words)}
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
    while not all(covered):
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
    total = q ** n  # 6561
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute neighbors
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = [None] * total
    for i in range(total):
        w = all_words[i]
        nbrs = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    nbrs.append(word_to_idx[tuple(w_new)])
        neighbors[i] = nbrs

    # Build adjacency matrix
    M = np.zeros((total, total), dtype=np.float32)
    for i in range(total):
        for j in neighbors[i]:
            M[i, j] = 1.0

    # Neighbors as numpy arrays for vectorized ops
    neighbors_arr = [np.array(neighbors[i], dtype=np.int32) for i in range(total)]

    best_code = None
    best_size = total

    def greedy_construct(uncovered_mask, code_set, cover_count, max_words=None):
        """Randomized greedy: pick word covering most uncovered, with random tie-breaking."""
        code_list = list(code_set)
        current_uncovered = uncovered_mask.copy()
        
        if max_words is not None:
            target = min(max_words, total)
        else:
            target = total

        while np.any(current_uncovered) and len(code_list) < target:
            if time.time() > deadline - 20:
                break
            scores = M @ current_uncovered
            # Mask out words already in code
            mask = np.ones(total, dtype=bool)
            for c in code_set:
                mask[c] = False
            scores_masked = np.where(mask, scores, -1)
            best_idx = int(np.argmax(scores_masked))
            if scores_masked[best_idx] <= 0:
                break
            code_list.append(best_idx)
            code_set.add(best_idx)
            cover_count[neighbors_arr[best_idx]] += 1
            current_uncovered[neighbors_arr[best_idx]] = 0.0
        
        return code_list

    def peel_reduce(code_list, code_set, cover_count):
        """Peeling: try to remove codeword c if the gap can be filled by one new codeword."""
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            random.shuffle(code_list)
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 20:
                    break
                c = code_list[i]
                # Try removing c
                cover_count[neighbors_arr[c]] -= 1
                uncovered_words = [j for j in neighbors_arr[c] if cover_count[j] == 0]
                
                if not uncovered_words:
                    # c was redundant
                    code_list.pop(i)
                    code_set.discard(c)
                    improved = True
                else:
                    # Check if a single codeword can cover all uncovered words
                    # Intersection of neighbors of all uncovered words
                    if len(uncovered_words) == 1:
                        candidates = set(neighbors[uncovered_words[0]]) - code_set
                    else:
                        candidates = set(neighbors[uncovered_words[0]])
                        for u in uncovered_words[1:]:
                            candidates &= set(neighbors[u])
                        candidates -= code_set
                    
                    if candidates:
                        # Pick best candidate: one that covers most other words
                        best_cand = None
                        best_score = -1
                        for cand in candidates:
                            # How many uncovered words does cand cover?
                            cnt = sum(1 for j in neighbors_arr[cand] if cover_count[j] == 0)
                            if cnt > best_score:
                                best_score = cnt
                                best_cand = cand
                        if best_cand is not None:
                            # Perform the swap: remove c, add best_cand
                            code_list[i] = best_cand
                            code_set.discard(c)
                            code_set.add(best_cand)
                            cover_count[neighbors_arr[best_cand]] += 1
                            improved = True
                        else:
                            cover_count[neighbors_arr[c]] += 1
                    else:
                        cover_count[neighbors_arr[c]] += 1
        return code_list, code_set, cover_count

    def full_reduction(code_list, code_set, cover_count):
        """Full reduction: peel + removal + swap."""
        # Phase 1: Peeling
        code_list, code_set, cover_count = peel_reduce(code_list, code_set, cover_count)
        
        # Phase 2: Pure removal
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            random.shuffle(code_list)
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 20:
                    break
                c = code_list[i]
                if np.all(cover_count[neighbors_arr[c]] > 1):
                    code_list.pop(i)
                    code_set.discard(c)
                    cover_count[neighbors_arr[c]] -= 1
                    improved = True
        return code_list, code_set, cover_count

    # Multiple restarts with different strategies
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        if time.time() > deadline - 25:
            break
            
        random.seed(restart * 12345 + 67890)
        
        code_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.float32)
        
        # Greedy construction
        code_list = greedy_construct(uncovered, code_set, cover_count)
        
        # Ensure full coverage
        if np.any(cover_count == 0):
            current_uncovered = (cover_count == 0).astype(np.float32)
            while np.any(current_uncovered) and time.time() < deadline - 25:
                scores = M @ current_uncovered
                mask = np.ones(total, dtype=bool)
                for c in code_set:
                    mask[c] = False
                scores_masked = np.where(mask, scores, -1)
                best_idx = int(np.argmax(scores_masked))
                if scores_masked[best_idx] <= 0:
                    break
                code_list.append(best_idx)
                code_set.add(best_idx)
                cover_count[neighbors_arr[best_idx]] += 1
                current_uncovered[neighbors_arr[best_idx]] = 0.0
        
        if len(code_list) >= best_size:
            continue
        
        # Reduction
        code_list, code_set, cover_count = full_reduction(code_list, code_set, cover_count)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)
            print(f"Restart {restart}: size {best_size}")

    if best_code is None:
        # Fallback: full greedy
        code_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.float32)
        code_list = greedy_construct(uncovered, code_set, cover_count)
        while np.any(cover_count == 0) and time.time() < deadline - 5:
            scores = M @ (cover_count == 0).astype(np.float32)
            mask = np.ones(total, dtype=bool)
            for c in code_set:
                mask[c] = False
            scores_masked = np.where(mask, scores, -1)
            best_idx = int(np.argmax(scores_masked))
            if scores_masked[best_idx] <= 0:
                break
            code_list.append(best_idx)
            code_set.add(best_idx)
            cover_count[neighbors_arr[best_idx]] += 1
        return [list(all_words[c]) for c in code_list]
    
    return [list(all_words[c]) for c in best_code]
