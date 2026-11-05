import time
import random
import itertools
import numpy as np

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
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
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
    radius = 1
    total = q ** n  # 6561

    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for return/cleanup

    # Precompute neighbors for all 6561 words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)

    # Precompute neighbors as lists of ints
    neighbors = [None] * total
    for idx in range(total):
        w = idx_to_word(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = list(w)
                    w_new[pos] = val
                    nbrs.append(word_to_idx(tuple(w_new)))
        neighbors[idx] = nbrs

    # Precompute neighbor sets for O(1) membership and intersection
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # Precompute distance-2 balls: words within Hamming distance <= 2 of each word
    # This is needed for incremental gain updates
    dist2_balls = [None] * total
    for idx in range(total):
        w = idx_to_word(idx)
        d2 = set()
        d2.add(idx)
        # Distance 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = list(w)
                    w_new[pos] = val
                    d2.add(word_to_idx(tuple(w_new)))
        # Distance 2
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                orig1 = w[pos1]
                orig2 = w[pos2]
                for val1 in range(q):
                    if val1 == orig1:
                        continue
                    for val2 in range(q):
                        if val2 == orig2:
                            continue
                        w_new = list(w)
                        w_new[pos1] = val1
                        w_new[pos2] = val2
                        d2.add(word_to_idx(tuple(w_new)))
        dist2_balls[idx] = list(d2)

    # Bit-parallel representation for fast gain calculation
    num_blocks = (total + 63) // 64  # 103
    neighbor_bits = [None] * total
    for i in range(total):
        blocks = [0] * num_blocks
        for nbr in neighbors[i]:
            blocks[nbr // 64] |= (1 << (nbr % 64))
        neighbor_bits[i] = blocks

    # Greedy construction with incremental gain updates using dist2_balls
    covered_mask = [0] * num_blocks
    cover_count = np.zeros(total, dtype=np.int32)

    code_list = []
    code_set = set()

    # Initialize gains: all 17
    gains = np.full(total, 17, dtype=np.int32)

    def is_covered(idx):
        return (covered_mask[idx // 64] >> (idx % 64)) & 1

    def mark_covered(idx):
        covered_mask[idx // 64] |= (1 << (idx % 64))

    # Greedy phase with incremental updates
    while time.time() < deadline - 60:
        # Check if all covered
        all_covered = True
        for b in range(num_blocks):
            if b < num_blocks - 1:
                if covered_mask[b] != ((1 << 64) - 1):
                    all_covered = False
                    break
            else:
                valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                if covered_mask[b] != valid_mask:
                    all_covered = False
                    break

        if all_covered:
            break

        # Find best candidate (max gain, not in code)
        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        # Add best_idx
        code_set.add(best_idx)
        code_list.append(best_idx)

        # Mark covered
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
            cover_count[nbr] += 1

        # Update gains incrementally: only words in dist2_balls[best_idx] are affected
        # For each word w in dist2_balls[best_idx] (and not in code):
        #   gains[w] decreases by |neighbors[w] ∩ neighbors[best_idx] ∩ newly_covered|
        # Since we just covered neighbors[best_idx], and before that some may have been covered:
        # The decrease for w is: count of nbr in neighbors[best_idx] that are in neighbors[w] and were NOT already covered before this step
        # Simpler: recompute gains[w] for w in dist2_balls[best_idx]
        for w in dist2_balls[best_idx]:
            if w in code_set:
                continue
            cnt = 0
            for nbr in neighbors[w]:
                if not is_covered(nbr):
                    cnt += 1
            gains[w] = cnt

    # Check if we've covered everything
    while time.time() < deadline - 45:
        all_covered = True
        for b in range(num_blocks):
            if b < num_blocks - 1:
                if covered_mask[b] != ((1 << 64) - 1):
                    all_covered = False
                    break
            else:
                valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                if covered_mask[b] != valid_mask:
                    all_covered = False
                    break

        if all_covered:
            break

        # Find best candidate
        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            cnt = 0
            for nbr in neighbors[i]:
                if not is_covered(nbr):
                    cnt += 1
            if cnt > best_gain:
                best_gain = cnt
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        code_set.add(best_idx)
        code_list.append(best_idx)

        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
            cover_count[nbr] += 1

        # Update gains incrementally
        for w in dist2_balls[best_idx]:
            if w in code_set:
                continue
            cnt = 0
            for nbr in neighbors[w]:
                if not is_covered(nbr):
                    cnt += 1
            gains[w] = cnt

    # Ensure full coverage: if not fully covered, add more greedily
    while time.time() < deadline - 40:
        all_covered = True
        for b in range(num_blocks):
            if b < num_blocks - 1:
                if covered_mask[b] != ((1 << 64) - 1):
                    all_covered = False
                    break
            else:
                valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                if covered_mask[b] != valid_mask:
                    all_covered = False
                    break

        if all_covered:
            break

        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            cnt = 0
            for nbr in neighbors[i]:
                if not is_covered(nbr):
                    cnt += 1
            if cnt > best_gain:
                best_gain = cnt
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        code_set.add(best_idx)
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
            cover_count[nbr] += 1

    # Now do local search to minimize size
    # Rebuild cover_count from scratch for accuracy
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    best_size = len(code_list)
    best_code = list(code_list)

    no_improvement_count = 0
    max_no_improvement = 30

    while time.time() < deadline - 3:
        # Phase 1: Try to remove words (most impactful)
        removed_any = False
        random.shuffle(code_list)

        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 8:
                break
            c = code_list[i]
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)

        if removed_any:
            no_improvement_count = 0
            continue

        # Phase 2: Try single swaps
        if len(code_list) > 440:
            improved_swap = False
            for _attempt in range(5):  # Try up to 5 random swaps per iteration
                if time.time() > deadline - 8:
                    break
                idx_c = random.randint(0, len(code_list) - 1)
                c = code_list[idx_c]

                # Temporarily remove
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1

                # Find uncovered words
                uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]

                if not uncovered_words:
                    # Safe removal
                    code_list.pop(idx_c)
                    code_set.discard(c)
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    improved_swap = True
                    break

                # Find replacement: must cover all uncovered words
                # Start with neighbors of first uncovered word, intersect with others
                if len(uncovered_words) <= 6:  # Only feasible for small sets
                    candidate_set = neighbor_sets[uncovered_words[0]].copy()
                    for u in uncovered_words[1:]:
                        candidate_set &= neighbor_sets[u]

                    candidate_set.discard(c)
                    candidate_set -= code_set

                    found_swap = False
                    for idx_cp in candidate_set:
                        # Verify it covers all uncovered
                        ok = True
                        for u in uncovered_words:
                            if u not in neighbor_sets[idx_cp]:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Perform swap
                        code_list[idx_c] = idx_cp
                        code_set.discard(c)
                        code_set.add(idx_cp)
                        for nbr in neighbors[idx_cp]:
                            cover_count[nbr] += 1
                        found_swap = True
                        no_improvement_count = 0
                        improved_swap = True
                        break

                    if not found_swap:
                        # Revert
                        for nbr in neighbors[c]:
                            cover_count[nbr] += 1

            if improved_swap:
                continue

        # Phase 3: 2-opt escape (remove two, add one) when stuck
        if no_improvement_count > max_no_improvement:
            if time.time() < deadline - 10:
                if len(code_list) >= 2:
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]

                    # Temporarily remove both
                    for nbr in neighbors[c1]:
                        cover_count[nbr] -= 1
                    for nbr in neighbors[c2]:
                        cover_count[nbr] -= 1

                    # Find uncovered words
                    all_uncovered = set()
                    for nbr in neighbors[c1]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    for nbr in neighbors[c2]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)

                    if not all_uncovered:
                        # Safe removal of both
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        no_improvement_count = 0
                        continue

                    # Find replacement that covers all uncovered words
                    ulist = list(all_uncovered)
                    if len(ulist) <= 8:
                        candidate_set = neighbor_sets[ulist[0]].copy()
                        for u in ulist[1:]:
                            candidate_set &= neighbor_sets[u]

                        candidate_set.discard(c1)
                        candidate_set.discard(c2)
                        candidate_set -= code_set

                        found = False
                        for idx_cp in candidate_set:
                            ok = True
                            for u in ulist:
                                if u not in neighbor_sets[idx_cp]:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            # Swap: remove c1, c2, add idx_cp
                            code_list.remove(c2)
                            code_list.remove(c1)
                            code_list.append(idx_cp)
                            code_set.discard(c1)
                            code_set.discard(c2)
                            code_set.add(idx_cp)
                            for nbr in neighbors[idx_cp]:
                                cover_count[nbr] += 1
                            found = True
                            no_improvement_count = 0
                            break

                        if not found:
                            # Revert
                            for nbr in neighbors[c1]:
                                cover_count[nbr] += 1
                            for nbr in neighbors[c2]:
                                cover_count[nbr] += 1
                            no_improvement_count += 1
                    else:
                        # Too many uncovered to find single replacement, revert
                        for nbr in neighbors[c1]:
                            cover_count[nbr] += 1
                        for nbr in neighbors[c2]:
                            cover_count[nbr] += 1
                        no_improvement_count += 1

        # Phase 4: Aggressive restart when very stuck
        if no_improvement_count > max_no_improvement * 3:
            if time.time() < deadline - 15:
                num_remove = max(1, len(code_list) // 8)
                to_remove = random.sample(code_list, min(num_remove, len(code_list)))
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)

                # Rebuild greedily
                while time.time() < deadline - 10:
                    all_covered = True
                    for idx in range(total):
                        if cover_count[idx] == 0:
                            all_covered = False
                            break
                    if all_covered:
                        break

                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = sum(1 for nbr in neighbors[idx] if cover_count[nbr] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx

                    if best_idx == -1 or best_count == 0:
                        break

                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    for nbr in neighbors[best_idx]:
                        cover_count[nbr] += 1

                no_improvement_count = 0

    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
