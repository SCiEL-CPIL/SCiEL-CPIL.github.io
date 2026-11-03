import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # radius = 1

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # ---------- Precompute balls ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # ---------- Inverse index: which centers cover each word ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Tetracode structured initializer ----------
    def structured_init():
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        # Greedy extension of the structured seed
        uncovered = (cover_count == 0)
        code_set = set(code_ints)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            code_set.add(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # ---------- Pure greedy set-cover fallback ----------
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        code_set = set()
        while covered_count < total and time.time() < deadline - 5:
            gains = ball_size - covered[balls].sum(axis=1)
            best_word = int(np.argmax(gains))
            if gains[best_word] == 0:
                break
            code.append(best_word)
            code_set.add(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # ---------- Choose the better starting point ----------
    code_struct, cc_struct = structured_init()

    def trim_inplace(code, cover_count, code_set):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    code_set_struct = set(code_struct)
    trim_inplace(code_struct, cc_struct, code_set_struct)

    code, cover_count, code_set = code_struct, cc_struct, code_set_struct

    if not code:
        code, cover_count = greedy_cover()
        code_set = set(code)
        trim_inplace(code, cover_count, code_set)

    # ---------- Aggressive trimming of redundant codewords ----------
    trim_inplace(code, cover_count, code_set)

    # ---------- Submodular-style local search: waste-minimizing swaps via inverse index ----------
    def local_search(time_budget=60.0):
        nonlocal code, cover_count, code_set
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(98765)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                # Submodular "waste" score
                best_swap = None
                best_waste = None
                best_new = -1
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    in_i = np.isin(bp_j, bp_i)
                    eff = cover_count[bp_j] - in_i.astype(np.int32)
                    waste = int(np.sum(eff >= 2))
                    new_cov = int(np.sum(eff == 0))
                    if best_waste is None or waste < best_waste or (waste == best_waste and new_cov > best_new):
                        best_waste = waste
                        best_new = new_cov
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    local_search(time_budget=60.0)
    trim_inplace(code, cover_count, code_set)

    # ---------- Cellular Automaton phase: coverage-deficit-driven global reorganization ----------
    def cellular_automaton(time_budget=55.0):
        nonlocal code, cover_count, code_set
        end_time = time.time() + time_budget
        
        # Build in-code indicator array
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code:
            in_code[c] = True
        
        # Precompute: for each word, the list of centers in its ball (inverse of balls)
        # Actually we already have centers_covering[w] = set of centers whose ball contains w
        # But we need: for each center c, which words are in its ball -> balls[c]
        # And for CA: for each word w, what's the coverage deficit = max(0, 1 - cover_count[w])
        
        # State: in_code[w] indicates whether word w is a center
        # Coverage of word w = sum of in_code[c] for c in centers_covering[w]
        # But we maintain cover_count separately for efficiency
        
        # CA update rule:
        # For each word w, compute deficit = max(0, 1 - cover_count[w])
        # A word w wants to become a center if deficit > 0 and it's not already
        # A word w wants to stop being a center if cover_count[w] >= 2 (redundant)
        # But we need to do this in a way that maintains coverage
        
        # Better CA rule:
        # Each word w has a "pressure" = cover_count[w] - 1 (positive = over-covered, negative = under-covered)
        # Centers with high pressure (over-covered) should be removed
        # Non-centers with negative pressure (under-covered) should be added
        # Use synchronous updates with a temperature-like acceptance
        
        # We'll do multiple CA steps
        ca_steps = 30
        step_time = time_budget / ca_steps
        
        for step in range(ca_steps):
            if time.time() > end_time:
                break
            
            step_start = time.time()
            
            # Compute pressure for each word
            pressure = cover_count.astype(np.float64) - 1.0  # >0: over-covered, <0: under-covered
            
            # Find candidates for removal: centers that are over-covered (all their ball points have cover_count >= 2)
            # Find candidates for addition: non-centers that cover under-covered points
            
            # For removal: a center c can be removed if all words in balls[c] have cover_count >= 2
            # For addition: a non-center c should be added if it covers many under-covered words
            
            # Compute gain for each potential addition
            under_covered = (cover_count <= 0).astype(np.float64)
            addition_gain = under_covered[balls].sum(axis=1)  # how many under-covered points each center covers
            
            # Compute redundancy for each center (can it be removed?)
            # A center c is removable if min(cover_count[balls[c]]) >= 2
            # But we need to be careful: removing one center might make another removable too
            # For CA: mark centers as "want to remove" if their ball is fully over-covered
            
            removable = np.zeros(total, dtype=np.bool_)
            for c in code:
                if (cover_count[balls[c]] >= 2).all():
                    removable[c] = True
            
            # Synchronous update:
            # 1. Remove centers that are marked removable (but not all at once - limit to avoid coverage loss)
            # 2. Add centers that have high addition_gain (but not too many)
            
            # Step 1: Remove redundant centers
            to_remove = [c for c in code if removable[c]]
            # Limit removals to maintain coverage
            max_removals = max(1, len(code) // 20)
            if len(to_remove) > max_removals:
                # Prioritize removal of centers with highest "redundancy" (min cover count in their ball)
                def redundancy_score(c):
                    return int(cover_count[balls[c]].min())
                to_remove.sort(key=redundancy_score, reverse=True)
                to_remove = to_remove[:max_removals]
            
            for c in to_remove:
                if c in code_set:
                    cover_count[balls[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    in_code[c] = False
            
            # Step 2: Add centers to cover under-covered points
            if np.any(cover_count <= 0):
                # Recompute addition gain after removals
                under_covered = (cover_count <= 0).astype(np.float64)
                addition_gain = under_covered[balls].sum(axis=1)
                
                # Find best candidates
                # Exclude current centers
                for c in code:
                    addition_gain[c] = -1
                
                max_additions = max(1, len(code) // 15)
                added = 0
                for _ in range(max_additions):
                    if time.time() > end_time:
                        break
                    best_c = int(np.argmax(addition_gain))
                    if addition_gain[best_c] <= 0:
                        break
                    code.append(best_c)
                    code_set.add(best_c)
                    in_code[best_c] = True
                    cover_count[balls[best_c]] += 1
                    addition_gain[best_c] = -1
                    added += 1
            
            # Trim after each CA step
            if time.time() < end_time - 2:
                trim_inplace(code, cover_count, code_set)
            
            elapsed = time.time() - step_start
            if elapsed > step_time * 2:
                # Running too slow, break
                break
        
        return code, cover_count, code_set

    cellular_automaton(time_budget=55.0)
    trim_inplace(code, cover_count, code_set)

    # ---------- Final local search polish ----------
    def local_search_polish(time_budget=20.0):
        nonlocal code, cover_count, code_set
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(12345)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                best_swap = None
                best_waste = None
                best_new = -1
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    in_i = np.isin(bp_j, bp_i)
                    eff = cover_count[bp_j] - in_i.astype(np.int32)
                    waste = int(np.sum(eff >= 2))
                    new_cov = int(np.sum(eff == 0))
                    if best_waste is None or waste < best_waste or (waste == best_waste and new_cov > best_new):
                        best_waste = waste
                        best_new = new_cov
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    local_search_polish(time_budget=20.0)
    trim_inplace(code, cover_count, code_set)

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result
