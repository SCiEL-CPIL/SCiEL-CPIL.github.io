"""Combined approach: tetracode seed + index-based local search with swap moves."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _tetracode(q, n):
    """Direct sum of ternary tetracode with all words on remaining coordinates."""
    if q != 3 or n < 4:
        # Fallback: just return a few random words
        return [list(random.choices(range(q), k=n)) for _ in range(10)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _word_to_idx(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _ball_indices_fast(idx, q, n, radius):
    """Return frozenset of indices within Hamming distance radius from idx."""
    if radius == 1:
        result = {idx}
        word = _idx_to_word(idx, q, n)
        for pos in range(n):
            pow_q = q ** pos
            digit = (idx // pow_q) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q
                    result.add(new_idx)
        return frozenset(result)
    else:
        # General case: enumerate
        word = _idx_to_word(idx, q, n)
        result = set()
        
        def gen(pos, current, dist):
            if dist > radius:
                return
            if pos == n:
                idx2 = 0
                for v in current:
                    idx2 = idx2 * q + v
                result.add(idx2)
                return
            current.append(word[pos])
            gen(pos + 1, current, dist)
            current.pop()
            for delta in range(1, q):
                current.append((word[pos] + delta) % q)
                gen(pos + 1, current, dist + 1)
                current.pop()
        
        gen(0, [], 0)
        return frozenset(result)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 6 and radius == 1:
        # Start with tetracode construction (81 words)
        initial_words = _tetracode(3, 6)
        code = [w for w in initial_words]
        code_set = set(tuple(w) for w in code)
        
        total = 3 ** 6
        # Precompute balls for all possible words
        all_balls = {}
        for idx in range(total):
            all_balls[idx] = _ball_indices_fast(idx, 3, 6, 1)
        
        # Convert code to indices
        code_indices = [_word_to_idx(w, 3, 6) for w in code]
        code_idx_set = set(code_indices)
        
        # Compute coverage
        coverage = [0] * total
        for ci in code_indices:
            for u in all_balls[ci]:
                coverage[u] += 1
        
        # Verify full coverage
        uncovered = [u for u in range(total) if coverage[u] == 0]
        
        # If not fully covered, add words greedily
        while uncovered and time.time() < deadline - 5.0:
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in code_idx_set:
                    continue
                gain = sum(1 for u in all_balls[c] if coverage[u] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1 or best_gain == 0:
                break
            code_indices.append(best_c)
            code_idx_set.add(best_c)
            for u in all_balls[best_c]:
                coverage[u] += 1
            uncovered = [u for u in range(total) if coverage[u] == 0]
        
        # Local search: try to remove redundant words and swap to reduce count
        rng = random.Random(42)
        best_code = code_indices[:]
        best_size = len(code_indices)
        
        temp = 1.0
        cool_rate = 0.9997
        min_temp = 0.05
        
        while time.time() < deadline - 2.0:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try remove
                if len(code_indices) > 0:
                    idx = rng.randint(0, len(code_indices) - 1)
                    c = code_indices[idx]
                    # Check if removable
                    removable = True
                    for u in all_balls[c]:
                        if coverage[u] == 1:
                            removable = False
                            break
                    if removable:
                        code_indices.pop(idx)
                        code_idx_set.discard(c)
                        for u in all_balls[c]:
                            coverage[u] -= 1
                        if len(code_indices) < best_size:
                            best_size = len(code_indices)
                            best_code = code_indices[:]
                        
            elif move_type < 0.85:
                # Try swap
                if len(code_indices) > 0:
                    idx = rng.randint(0, len(code_indices) - 1)
                    old_c = code_indices[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_idx_set:
                        continue
                    
                    # Check if swap is valid
                    valid = True
                    for u in all_balls[old_c]:
                        if coverage[u] == 1 and u not in all_balls[new_c]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        for u in all_balls[old_c]:
                            coverage[u] -= 1
                        for u in all_balls[new_c]:
                            coverage[u] += 1
                        code_indices[idx] = new_c
                        code_idx_set.discard(old_c)
                        code_idx_set.add(new_c)
                        
                        if len(code_indices) < best_size:
                            best_size = len(code_indices)
                            best_code = code_indices[:]
                        
            else:
                # Try add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_idx_set:
                    if rng.random() < min(1.0, 0.5 * temp):
                        code_indices.append(new_c)
                        code_idx_set.add(new_c)
                        for u in all_balls[new_c]:
                            coverage[u] += 1
            
            temp *= cool_rate
        
        # Use best code found
        final_indices = best_code
        
        # Final cleanup: remove any redundant words
        # Recompute coverage
        coverage2 = [0] * total
        for ci in final_indices:
            for u in all_balls[ci]:
                coverage2[u] += 1
        
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(final_indices)):
                c = final_indices[i]
                removable = True
                for u in all_balls[c]:
                    if coverage2[u] == 1:
                        removable = False
                        break
                if removable:
                    final_indices.pop(i)
                    for u in all_balls[c]:
                        coverage2[u] -= 1
                    changed = True
                    break
        
        # Convert back to words
        result = [_idx_to_word(ci, 3, 6) for ci in final_indices]
        return result
    
    else:
        # General case: greedy cover
        total = q ** n
        all_balls = {}
        for idx in range(total):
            all_balls[idx] = _ball_indices_fast(idx, q, n, radius)
        
        uncovered = set(range(total))
        code_indices = []
        code_idx_set = set()
        
        while uncovered and time.time() < deadline - 10.0:
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in code_idx_set:
                    continue
                gain = len(all_balls[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1 or best_gain == 0:
                break
            code_indices.append(best_c)
            code_idx_set.add(best_c)
            uncovered -= all_balls[best_c]
        
        # Local search
        coverage = [0] * total
        for ci in code_indices:
            for u in all_balls[ci]:
                coverage[u] += 1
        
        rng = random.Random(42)
        best_code = code_indices[:]
        best_size = len(code_indices)
        
        temp = 1.0
        cool_rate = 0.9997
        min_temp = 0.05
        
        while time.time() < deadline - 2.0:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.5:
                if len(code_indices) > 0:
                    idx = rng.randint(0, len(code_indices) - 1)
                    c = code_indices[idx]
                    removable = True
                    for u in all_balls[c]:
                        if coverage[u] == 1:
                            removable = False
                            break
                    if removable:
                        code_indices.pop(idx)
                        code_idx_set.discard(c)
                        for u in all_balls[c]:
                            coverage[u] -= 1
                        if len(code_indices) < best_size:
                            best_size = len(code_indices)
                            best_code = code_indices[:]
            
            elif move_type < 0.85:
                if len(code_indices) > 0:
                    idx = rng.randint(0, len(code_indices) - 1)
                    old_c = code_indices[idx]
                    new_c = rng.randint(0, total - 1)
                    if new_c == old_c or new_c in code_idx_set:
                        continue
                    valid = True
                    for u in all_balls[old_c]:
                        if coverage[u] == 1 and u not in all_balls[new_c]:
                            valid = False
                            break
                    if valid:
                        for u in all_balls[old_c]:
                            coverage[u] -= 1
                        for u in all_balls[new_c]:
                            coverage[u] += 1
                        code_indices[idx] = new_c
                        code_idx_set.discard(old_c)
                        code_idx_set.add(new_c)
                        if len(code_indices) < best_size:
                            best_size = len(code_indices)
                            best_code = code_indices[:]
            
            else:
                new_c = rng.randint(0, total - 1)
                if new_c not in code_idx_set:
                    if rng.random() < min(1.0, 0.5 * temp):
                        code_indices.append(new_c)
                        code_idx_set.add(new_c)
                        for u in all_balls[new_c]:
                            coverage[u] += 1
            
            temp *= cool_rate
        
        final_indices = best_code
        
        # Final cleanup
        coverage2 = [0] * total
        for ci in final_indices:
            for u in all_balls[ci]:
                coverage2[u] += 1
        
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(final_indices)):
                c = final_indices[i]
                removable = True
                for u in all_balls[c]:
                    if coverage2[u] == 1:
                        removable = False
                        break
                if removable:
                    final_indices.pop(i)
                    for u in all_balls[c]:
                        coverage2[u] -= 1
                    changed = True
                    break
        
        result = [_idx_to_word(ci, q, n) for ci in final_indices]
        return result
# EVOLVE-BLOCK-END
