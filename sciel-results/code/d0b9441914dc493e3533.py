import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 112.0  # Safety margin for 120s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15
        FULL_MASK = (1 << N) - 1

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute cover_mask: bitset of words covered by each codeword
        # Also precompute cover_map: for each word, list of codewords covering it
        cover_mask = [0] * N
        cover_map = [[] for _ in range(N)]

        for i in range(N):
            digits = int_to_base3(i, n)
            mask = 1 << i
            cover_map[i].append(i)
            
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        w = base3_to_int(new_digits)
                        mask |= (1 << w)
                        cover_map[w].append(i)
            
            cover_mask[i] = mask

        # Precompute bit indices for each cover_mask for faster iteration
        # cover_bits[i] = list of word indices covered by codeword i
        cover_bits = []
        for i in range(N):
            bits = []
            mask = cover_mask[i]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                bits.append(idx)
                mask ^= lsb
            cover_bits.append(bits)

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with random tie-breaking."""
            rng = random.Random(seed)
            uncovered = FULL_MASK
            code = []
            in_code = [False] * N
            
            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    uncovered &= ~cover_mask[c]

            while uncovered:
                best_c = -1
                best_gain = 0
                candidates = []
                for c in range(N):
                    if not in_code[c]:
                        gain = (cover_mask[c] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [c]
                        elif gain == best_gain and gain > 0:
                            candidates.append(c)
                
                if not candidates:
                    break
                    
                c = rng.choice(candidates)
                code.append(c)
                in_code[c] = True
                uncovered &= ~cover_mask[c]

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in cover_bits[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in cover_bits[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in cover_bits[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, and 2-swap moves using bitsets."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            # Initialize cover_count
            cover_count = [0] * N
            for c in code_list:
                for w in cover_bits[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                # 1. Try Deletion and 1-Swap
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    
                    # Check if deletion is safe
                    uncov_list = []
                    for w in cover_bits[c_remove]:
                        if cover_count[w] == 1:
                            uncov_list.append(w)

                    if not uncov_list:
                        # Safe to delete
                        for w in cover_bits[c_remove]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-Swap: Find c_add such that cover_mask[c_add] covers uncov_list
                    # Sort uncov_list by rarity to speed up intersection
                    uncov_sorted = sorted(uncov_list, key=lambda w: len(cover_map[w]))
                    cand_pool = None
                    for w in uncov_sorted:
                        cands = [c for c in cover_map[w] if not in_code[c] and c != c_remove]
                        if cand_pool is None:
                            cand_pool = set(cands)
                        else:
                            cand_pool.intersection_update(cands)
                        if not cand_pool:
                            break
                    
                    if cand_pool:
                        # Pick one that maximizes overlap with current code (redundancy) or just random
                        # To be fast, pick one that minimizes "waste" = covers words already well-covered
                        # For now, random pick from pool
                        found = rng.choice(list(cand_pool))
                        
                        # Execute swap
                        for w in cover_bits[c_remove]:
                            cover_count[w] -= 1
                        in_code[c_remove] = False
                        
                        code_list[i] = found
                        in_code[found] = True
                        for w in cover_bits[found]:
                            cover_count[w] += 1
                            
                        improved = True
                        break

                # 2. Try 2-Swap: Remove c1, c2, add c3
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    # Sample a subset of pairs if too many
                    # n_codes ~ 186, pairs ~ 17k. This is fine.
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            
                            # Calculate uncovered words
                            uncov_list = []
                            mask1 = cover_mask[c_rem1]
                            mask2 = cover_mask[c_rem2]
                            
                            # Words covered by c_rem1
                            for w in cover_bits[c_rem1]:
                                if cover_count[w] == 1:
                                    uncov_list.append(w)
                                elif cover_count[w] == 2:
                                    if mask2 & (1 << w):
                                        uncov_list.append(w)
                            
                            # Words covered only by c_rem2 (not c_rem1)
                            # If count is 1, it becomes 0.
                            # If count is 2, the other coverer is not c_rem1 (since w not in mask1), so count becomes 1.
                            for w in cover_bits[c_rem2]:
                                if not (mask1 & (1 << w)):
                                    if cover_count[w] == 1:
                                        uncov_list.append(w)
                            
                            # Remove duplicates
                            if uncov_list:
                                uncov_list = list(set(uncov_list))

                            if not uncov_list:
                                # Safe to remove both?
                                # This implies we can reduce size by 2.
                                for w in cover_bits[c_rem1]:
                                    cover_count[w] -= 1
                                for w in cover_bits[c_rem2]:
                                    cover_count[w] -= 1
                                
                                # Pop j2 first if j2 > i2
                                code_list.pop(j2)
                                code_list.pop(i2)
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                improved = True
                                break
                            
                            # Find c_add covering uncov_list
                            uncov_sorted = sorted(uncov_list, key=lambda w: len(cover_map[w]))
                            cand_pool = None
                            for w in uncov_sorted:
                                cands = [c for c in cover_map[w] if not in_code[c] and c != c_rem1 and c != c_rem2]
                                if cand_pool is None:
                                    cand_pool = set(cands)
                                else:
                                    cand_pool.intersection_update(cands)
                                if not cand_pool:
                                    break
                            
                            if cand_pool:
                                found2 = rng.choice(list(cand_pool))
                                
                                # Execute 2-for-1
                                for w in cover_bits[c_rem1]:
                                    cover_count[w] -= 1
                                for w in cover_bits[c_rem2]:
                                    cover_count[w] -= 1
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                
                                # Replace c_rem2 with found2
                                code_list[j2] = found2
                                in_code[found2] = True
                                for w in cover_bits[found2]:
                                    cover_count[w] += 1
                                
                                code_list.pop(i2)
                                improved = True
                                break
                            
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        num_restarts = 150
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
            
            if time.time() > deadline - 5:
                continue
                
            ls_time = min(deadline - 3, time.time() + 2.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts
        if best_code is not None:
            for kick_iter in range(20):
                if time.time() > deadline - 8:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(5, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                
                if time.time() > deadline - 5:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 3, time.time() + 2.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 2:
            ls_code = local_search(best_code, 42, deadline - 1)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
