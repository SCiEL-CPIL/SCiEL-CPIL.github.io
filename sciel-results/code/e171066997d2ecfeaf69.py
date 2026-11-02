import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute place values
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball as bitmask (Python int) for each word
    ball_mask = [0] * total
    for i in range(total):
        mask = 1 << i
        digits = []
        temp = i
        for j in range(n - 1, -1, -1):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                mask |= (1 << new_idx)
        ball_mask[i] = mask

    # Precompute ball as list for iteration
    ball_list = [None] * total
    for i in range(total):
        lst = [i]
        digits = []
        temp = i
        for j in range(n - 1, -1, -1):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                lst.append(new_idx)
        ball_list[i] = lst

    def greedy_build(seed):
        rng = random.Random(seed)
        covered_mask = 0
        code = []
        candidates = list(range(total))
        rng.shuffle(candidates)
        for c in candidates:
            new_cov = bin(ball_mask[c] & ~covered_mask).count('1')
            if new_cov > 0:
                code.append(c)
                covered_mask |= ball_mask[c]
                if covered_mask == (1 << total) - 1:
                    break
        return code

    # Multiple greedy restarts
    best_initial = None
    best_initial_size = float('inf')
    for seed in range(20):
        if time.time() > deadline - 60.0:
            break
        init_code = greedy_build(seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = init_code

    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)

    # Count coverage using array
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        for w in ball_list[c]:
            cnt[w] += 1

    def run_delete_phase(code_set, cnt_arr):
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_list[c]
                redundant = True
                for w in ball:
                    if cnt_arr[w] < 2:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for w in ball:
                        cnt_arr[w] -= 1
                    removed += 1
                    changed = True
        return removed

    def try_swap(c_remove, c_add, code_set, cnt_arr):
        if c_add in code_set or c_remove == c_add:
            return False
        ball_rem = ball_list[c_remove]
        ball_add = ball_list[c_add]
        add_set = set(ball_add)
        for w in ball_rem:
            if w not in add_set and cnt_arr[w] < 2:
                return False
        code_set.remove(c_remove)
        code_set.add(c_add)
        for w in ball_rem:
            cnt_arr[w] -= 1
        for w in ball_add:
            cnt_arr[w] += 1
        return True

    def get_valid_swaps(c_remove, code_set, cnt_arr):
        ball_rem = ball_list[c_remove]
        unique_words = [w for w in ball_rem if cnt_arr[w] == 1]
        if not unique_words:
            return None
        valid_mask = ball_mask[unique_words[0]]
        for u in unique_words[1:]:
            valid_mask &= ball_mask[u]
            if valid_mask == 0:
                return []
        valid_list = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            if idx not in code_set:
                valid_list.append(idx)
            vm ^= lsb
        return valid_list

    def compress_two_to_one(code_set, cnt_arr, rng, deadline):
        """
        Attempt to replace two codewords c1, c2 with one new codeword c3.
        This is the "bit-parallel iterative compression" approach.
        """
        if len(code_set) < 2:
            return False
        
        # Precompute coverage by others
        full_cover = 0
        for c in code_set:
            full_cover |= ball_mask[c]
        
        # Randomly pick pairs
        code_list = list(code_set)
        attempts = 0
        while attempts < 50 and time.time() < deadline - 3.0:
            attempts += 1
            idx1 = rng.randrange(len(code_list))
            idx2 = rng.randrange(len(code_list) - 1)
            if idx2 >= idx1:
                idx2 += 1
            c1 = code_list[idx1]
            c2 = code_list[idx2]
            
            # What is covered by c1 and c2 but not by others?
            others_cover = full_cover & ~(ball_mask[c1] | ball_mask[c2])
            uncovered = (ball_mask[c1] | ball_mask[c2]) & ~others_cover
            
            if uncovered == 0:
                # Both are redundant together
                code_set.remove(c1)
                code_set.remove(c2)
                for w in ball_list[c1]:
                    cnt_arr[w] -= 1
                for w in ball_list[c2]:
                    cnt_arr[w] -= 1
                return True
            elif bin(uncovered).count('1') <= 8:
                # Try to find a single codeword that covers all uncovered words
                # Search in the neighborhood of the uncovered words
                found = None
                # Sample from the union of balls of uncovered words
                sample_set = set()
                for w in range(total):
                    if (uncovered >> w) & 1:
                        sample_set.update(ball_list[w])
                
                # Try random samples
                for _ in range(100):
                    if time.time() > deadline - 3.0:
                        break
                    if not sample_set:
                        break
                    cand = rng.choice(list(sample_set))
                    if cand in code_set:
                        continue
                    if (ball_mask[cand] & uncovered) == uncovered:
                        found = cand
                        break
                
                if found is not None:
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(found)
                    for w in ball_list[c1]:
                        cnt_arr[w] -= 1
                    for w in ball_list[c2]:
                        cnt_arr[w] -= 1
                    for w in ball_list[found]:
                        cnt_arr[w] += 1
                    return True
        
        return False

    # Local search
    no_improve = 0
    max_no_improve = 15
    rng = random.Random(123)

    while time.time() < deadline - 5.0:
        # Delete phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            no_improve = 0
        else:
            no_improve += 1

        # Compression phase (2-for-1)
        if len(current_code) >= 2 and time.time() < deadline - 5.0:
            if compress_two_to_one(current_code, cnt, rng, deadline):
                removed = run_delete_phase(current_code, cnt)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                no_improve = 0

        # Swap phase
        swaps_done = 0
        max_swaps = 400
        while swaps_done < max_swaps and time.time() < deadline - 5.0:
            swaps_done += 1
            if not current_code:
                break
            c = rng.choice(list(current_code))
            valid = get_valid_swaps(c, current_code, cnt)
            if valid is None:
                c_prime = rng.randint(0, total - 1)
                attempts = 0
                while c_prime in current_code and attempts < 100:
                    c_prime = rng.randint(0, total - 1)
                    attempts += 1
                if c_prime in current_code:
                    continue
            elif len(valid) == 0:
                no_improve += 1
                continue
            else:
                c_prime = rng.choice(valid)

            if try_swap(c, c_prime, current_code, cnt):
                removed = run_delete_phase(current_code, cnt)
                if removed > 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # Restart if stuck
        if no_improve > max_no_improve and time.time() < deadline - 20.0:
            seed = int(time.time() * 1000) % 100000
            init_code = greedy_build(seed)
            if len(init_code) < len(current_code) + 3:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    for w in ball_list[c]:
                        cnt[w] += 1
                no_improve = 0

        if best_size <= 160:
            break

    # Final polish
    if time.time() < deadline - 3.0:
        changed = True
        iters = 0
        while changed and time.time() < deadline - 2.0 and iters < 15:
            iters += 1
            changed = False
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                changed = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
            # Try all swaps
            for c in list(current_code):
                if time.time() > deadline - 2.0:
                    break
                valid = get_valid_swaps(c, current_code, cnt)
                if valid is None:
                    continue
                if not valid:
                    continue
                for cp in valid:
                    if time.time() > deadline - 2.0:
                        break
                    if try_swap(c, cp, current_code, cnt):
                        removed = run_delete_phase(current_code, cnt)
                        if removed > 0:
                            changed = True
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code = list(current_code)
                        else:
                            current_code.remove(cp)
                            current_code.add(c)
                            for w in ball_list[cp]:
                                cnt[w] -= 1
                            for w in ball_list[c]:
                                cnt[w] += 1

    # Convert to output format
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
