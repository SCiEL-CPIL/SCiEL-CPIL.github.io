import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using structured tetracode product initialization,
    vectorized greedy extension, aggressive trimming, and vectorized swap local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

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

    # Precompute balls
    balls = np.empty((total_words, ball_size), dtype=np.int32)
    for w in range(total_words):
        d = int_to_digits(w, n, q)
        neighbors = [w]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls[w] = neighbors

    # Precompute inverse index
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)
    for w in range(total_words):
        if inv_ball[w] is not None:
            inv_ball[w] = np.array(inv_ball[w], dtype=np.int32)

    # Structured tetracode product initialization
    def structured_init():
        # Tetracode: (3,3) linear code over GF(3)
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
            
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            # Mask out already chosen centers to avoid re-selection
            gains[code_ints] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # Greedy fallback if structured fails or for diversity
    def greedy_init():
        covered = np.zeros(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        while np.any(~covered) and time.time() < deadline - 5:
            gains = (~covered)[balls].sum(axis=1)
            best_center = int(np.argmax(gains))
            if gains[best_center] == 0:
                break
            code.append(best_center)
            cover_count[balls[best_center]] += 1
            covered[balls[best_center]] = True
        return code, cover_count

    # Select best initial code
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_init()
    
    code_set = set(code)

    def trim_code():
        nonlocal code, cover_count, code_set
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.all(cover_count[balls[c]] > 1):
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    trim_code()

    # Vectorized Local Search
    def local_search(seed, time_budget=60.0):
        nonlocal code, cover_count, code_set
        random.seed(seed)
        end_time = time.time() + time_budget
        improved = True
        temp = 1.0
        
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > end_time:
                    return
                i = code[idx]
                bi = balls[i]
                
                # Check if redundant
                unique_pts = bi[cover_count[bi] == 1]
                if len(unique_pts) == 0:
                    code.pop(idx)
                    cover_count[bi] -= 1
                    code_set.discard(i)
                    improved = True
                    break
                
                # Find candidates covering all unique points
                # Use set intersection on numpy arrays
                candidate_set = None
                for w in unique_pts:
                    cw = inv_ball[w]
                    if candidate_set is None:
                        candidate_set = set(cw.tolist())
                    else:
                        candidate_set &= set(cw.tolist())
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                # Exclude current code
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                
                candidates_arr = np.array(candidates, dtype=np.int32)
                
                # Vectorized scoring:
                # We want to minimize "waste" or maximize "efficiency".
                # A good swap reduces the number of words that become over-covered unnecessarily,
                # or covers unique words of i.
                # Let's score by: sum over j in candidates of (number of words in ball[j] 
                # that are currently covered by exactly 1 codeword (which is i) 
                # minus number of words in ball[j] that are covered by >= 2 codewords).
                
                # Actually, a simpler heuristic: 
                # Gain = count of words in ball[j] where cover_count == 1 (and that word is in ball[i])
                # Loss = count of words in ball[j] where cover_count >= 2
                # We want high Gain, low Loss.
                
                # Let's compute for all candidates at once if possible, or loop if too many.
                # Given the size (6561), looping over candidates is okay if candidates are few.
                # But vectorizing is better.
                
                # For each candidate j, score = sum( (cover_count[balls[j]] == 1) & (balls[j] is in bi) ) - sum( cover_count[balls[j]] >= 2 )
                # This is hard to vectorize across j directly without a loop over j or large memory.
                # Let's stick to a fast Python loop over candidates, but use numpy for the inner calculation.
                
                best_swap = None
                best_score = -float('inf')
                
                # Shuffle candidates for diversity
                random.shuffle(candidates)
                # Limit candidates to avoid excessive time per swap attempt
                n_check = min(50, len(candidates))
                
                for j in candidates[:n_check]:
                    if time.time() > end_time:
                        return
                    bj = balls[j]
                    # Words in bj that are uniquely covered by i (cover_count == 1)
                    # Note: cover_count[bi] == 1 implies i is the only cover.
                    # So we check if cover_count[bj] == 1.
                    unique_gain = np.sum(cover_count[bj] == 1)
                    # Over-cover loss
                    over_loss = np.sum(cover_count[bj] >= 2)
                    
                    # Score: we prefer swaps that pick up unique words and don't create too much redundancy
                    score = unique_gain * 2 - over_loss
                    
                    if score > best_score:
                        best_score = score
                        best_swap = j

                if best_swap is not None:
                    j = best_swap
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[bi] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with multiple seeds and time splits
    # Split remaining time into chunks
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        # First pass
        local_search(42, time_budget=min(60, remaining_time * 0.4))
        trim_code()
        
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            # Second pass with different seed
            local_search(123, time_budget=min(60, remaining_time * 0.4))
            trim_code()
            
            remaining_time = deadline - time.time() - 5
            if remaining_time > 10:
                # Third pass
                local_search(789, time_budget=min(60, remaining_time * 0.4))
                trim_code()

    # Final trim
    trim_code()

    result = [int_to_digits(c, n, q) for c in code]
    return result
