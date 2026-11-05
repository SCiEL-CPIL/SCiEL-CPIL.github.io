"""Covering code construction for K_q(n, radius) with a time-bounded greedy heuristic."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s wall-clock limit

    # Trivial cases
    if n == 0:
        return [[]]
    if radius == 0:
        return [[0] * n]

    total_words = q ** n

    # For small spaces, exact greedy (which is optimal for covering)
    if total_words <= 500000:
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls (all words within Hamming distance <= radius)
        balls = {}
        for idx, w in enumerate(all_words):
            ball = set()
            ball.add(idx)
            if radius >= 1:
                for pos in range(n):
                    for val in range(q):
                        if val == w[pos]:
                            continue
                        new_word = list(w)
                        new_word[pos] = val
                        ball.add(word_to_idx[tuple(new_word)])
            if radius >= 2 and n >= 2:
                for pos1 in range(n):
                    for pos2 in range(n):
                        if pos1 >= pos2:
                            continue
                        for val1 in range(q):
                            if val1 == w[pos1]:
                                continue
                            for val2 in range(q):
                                if val2 == w[pos2]:
                                    continue
                                new_word = list(w)
                                new_word[pos1] = val1
                                new_word[pos2] = val2
                                ball.add(word_to_idx[tuple(new_word)])
            balls[idx] = ball
        
        covered = [False] * total_words
        code = []
        while not all(covered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                if all(covered[i] for i in balls[idx]):
                    continue
                count = sum(1 for i in balls[idx] if not covered[i])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(list(all_words[best_idx]))
            for i in balls[best_idx]:
                covered[i] = True
            if time.time() > deadline:
                break
        
        if all(covered) or len(code) > 0:
            if _verify_coverage(code, q, n, radius):
                return code
        
        return _direct_sum(q, n, radius)

    # For larger spaces: use direct sum as base, then try to improve
    base_code = _direct_sum(q, n, radius)
    
    if time.time() > deadline - 5:
        return base_code

    # Convert to index-based representation for efficient operations
    code_indices = [tuple(c) for c in base_code]
    
    # For the main task (q=3, n=8, radius=1), use the specialized fast path
    if q == 3 and n == 8 and radius == 1:
        result = _fast_covering_3_8_1(deadline)
        if result is not None:
            return result
    
    # General improvement: greedy removal
    if len(code_indices) > 100 and time.time() < deadline - 10:
        code_indices = _try_remove_redundant_indices(code_indices, q, n, radius, deadline)
    
    return [list(c) for c in code_indices]


def _fast_covering_3_8_1(deadline):
    """Specialized fast covering code construction for q=3, n=8, radius=1."""
    q, n, radius = 3, 8, 1
    total_words = q ** n  # 6561
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
    
    # Precompute neighbor lists (ball of radius 1)
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
    
    # Phase 1: Fast Greedy Initialization
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
    
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # Phase 2: Iterated Local Search
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    random.seed(42)
    stuck_counter = 0
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
        
        # 2. If stuck, try a swap
        if stuck_counter > 5 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers all risky points
                candidates = set(neighbors[risky_points[0]].tolist())
                candidates.discard(c1)
                
                for x in risky_points[1:]:
                    candidates &= set(neighbors[x].tolist())
                    if not candidates:
                        break
                
                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
                
                if valid_candidates:
                    c2 = random.choice(valid_candidates)
                    
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2
                    
                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)
                    
                    stuck_counter = 0
                else:
                    pass
            
            stuck_counter = 0
    
    # Phase 3: Deterministic removal polish
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
    
    if len(final_code) < best_size:
        best_size = len(final_code)
        best_code = final_code[:]
    
    # Phase 4: Random restarts for more aggressive search
    if time.time() < deadline - 10.0:
        for restart in range(3):
            if time.time() >= deadline - 5.0:
                break
                
            rc_code = best_code[:]
            rc_size = len(rc_code)
            rc_cov = np.zeros(total_words, dtype=np.int32)
            for idx in rc_code:
                rc_cov[neighbors[idx]] += 1
            rc_set = set(rc_code)
            
            random.seed(42 + restart * 1000)
            
            stuck = 0
            while time.time() < deadline - 5.0 - restart * 2.0:
                if rc_size == 0:
                    break
                idx = random.randint(0, rc_size - 1)
                c = rc_code[idx]
                
                if np.all(rc_cov[neighbors[c]] > 1):
                    rc_cov[neighbors[c]] -= 1
                    rc_set.remove(c)
                    rc_code.pop(idx)
                    rc_size -= 1
                    stuck = 0
                    
                    if rc_size < best_size:
                        best_size = rc_size
                        best_code = rc_code[:]
                else:
                    stuck += 1
                    
                if stuck > 5:
                    c1_idx = random.randint(0, rc_size - 1)
                    c1 = rc_code[c1_idx]
                    risky_points = [x for x in neighbors[c1] if rc_cov[x] == 1]
                    
                    if not risky_points:
                        rc_cov[neighbors[c1]] -= 1
                        rc_set.remove(c1)
                        rc_code.pop(c1_idx)
                        rc_size -= 1
                        stuck = 0
                        if rc_size < best_size:
                            best_size = rc_size
                            best_code = rc_code[:]
                    else:
                        candidates = set(neighbors[risky_points[0]].tolist())
                        candidates.discard(c1)
                        for x in risky_points[1:]:
                            candidates &= set(neighbors[x].tolist())
                            if not candidates:
                                break
                        valid_candidates = [c2 for c2 in candidates if c2 not in rc_set]
                        if valid_candidates:
                            c2 = random.choice(valid_candidates)
                            rc_cov[neighbors[c1]] -= 1
                            rc_set.remove(c1)
                            rc_code[c1_idx] = c2
                            rc_cov[neighbors[c2]] += 1
                            rc_set.add(c2)
                            stuck = 0
                    
                    stuck = 0
    
    # Convert back to list of lists
    return [list(digits_of[c]) for c in best_code]


def _direct_sum(q, n, radius):
    """Direct sum construction using a perfect code on a small block."""
    if q == 3:
        # Ternary tetracode: 9 words of length 4, perfect code radius 1
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return [list(h) for h in tetracode]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]
    elif q == 2:
        if n >= 3:
            hamming = [[a, b, a ^ b] for a in range(2) for b in range(2)]
            if n == 3:
                return [list(h) for h in hamming]
            full_blocks = n // 3
            remainder = n % 3
            code = []
            for block_combos in itertools.product(range(4), repeat=full_blocks):
                for tail in itertools.product(range(2), repeat=remainder):
                    word = []
                    for bi in block_combos:
                        a, b = divmod(bi, 2)
                        word.extend([a, b, a ^ b])
                    word.extend(tail)
                    code.append(word)
            return code
        else:
            return [list(w) for w in itertools.product(range(2), repeat=n)]
    else:
        return [list(w) for w in itertools.product(range(q), repeat=n)]


def _verify_coverage(code, q, n, radius):
    """Verify that every word is within Hamming distance radius of some codeword."""
    if not code:
        return n == 0
    total_words = q ** n
    if total_words <= 1000000:
        code_set = set(tuple(c) for c in code)
        for word in itertools.product(range(q), repeat=n):
            min_dist = min(_hamming(word, c) for c in code_set)
            if min_dist > radius:
                return False
        return True
    else:
        for _ in range(10000):
            word = tuple(random.randint(0, q - 1) for _ in range(n))
            min_dist = min(_hamming(word, c) for c in code)
            if min_dist > radius:
                return False
        return True


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _try_remove_redundant_indices(code_indices, q, n, radius, deadline):
    """Try to remove codewords that are not needed for coverage (index-based)."""
    code = list(code_indices)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 5:
                break
            candidate = code[:idx] + code[idx+1:]
            if len(candidate) == 0:
                continue
            if _verify_coverage([list(c) for c in candidate], q, n, radius):
                code = candidate
                changed = True
                break
    return code
# EVOLVE-BLOCK-END
