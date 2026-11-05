import numpy as np
import random
import time

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

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

        # Precompute balls (neighbors including self)
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute balls as numpy arrays for vectorized operations
        balls_np = np.array(balls, dtype=np.int32)
        
        # Precompute cover_map as list of numpy arrays
        cover_map_np = [np.array(cm, dtype=np.int32) for cm in cover_map]

        def greedy_cover(seed):
            """Greedy set cover with numpy-accelerated gain computation."""
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            code = []
            uncovered_count = N

            while uncovered_count > 0:
                # Vectorized gain computation
                gains = balls_np[:, :].reshape(N, -1)
                # Count uncovered neighbors for each candidate
                gains = np.sum(~covered[balls_np], axis=1)
                max_gain = gains.max()
                candidates = np.where(gains == max_gain)[0]
                c = int(rng.choice(candidates))
                code.append(c)

                # Update covered and uncovered_count
                ball_c = balls_np[c]
                newly_covered = ball_c[~covered[ball_c]]
                covered[ball_c] = True
                uncovered_count -= len(newly_covered)

                # Update gains: for each newly covered word, decrement gain of its coverers
                for w in newly_covered:
                    for c2 in cover_map_np[w]:
                        gains[c2] -= 1

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        # Precompute ball sets for fast membership testing
        ball_sets = [set(b) for b in balls]
        # Precompute which words are covered by which codewords (inverse)
        # For swap: need to find codeword c_add such that all uncovered words are in ball[c_add]

        def local_search(code, seed, time_limit):
            """Local search with optimized deletion and swap moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            # Use numpy for cover_count
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[balls_np[c]] += 1

            code_set = set(code_list)

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    ball_rm = balls_np[c_remove]
                    
                    # Try deletion: only check words in ball_rm
                    cover_count[ball_rm] -= 1
                    # A word is uncovered if cover_count[w] == 0 AND w was in ball_rm
                    # (words not in ball_rm still have cover_count >= 1 since c_remove didn't cover them)
                    uncovered_in_ball = ball_rm[cover_count[ball_rm] == 0]
                    
                    if len(uncovered_in_ball) == 0:
                        code_list.pop(i)
                        code_set.remove(c_remove)
                        improved = True
                        break

                    # Try swap: find c_add that covers all uncovered_in_ball
                    found = -1
                    # For efficiency, check candidates that cover the first uncovered word
                    first_uncovered = uncovered_in_ball[0]
                    candidates = cover_map_np[first_uncovered]
                    for c_add in candidates:
                        if c_add == c_remove or c_add in code_set:
                            continue
                        # Check if c_add covers all uncovered words
                        ball_add = balls_np[c_add]
                        # All uncovered words must be in ball_add
                        if np.all(np.isin(uncovered_in_ball, ball_add)):
                            found = c_add
                            break
                    
                    if found >= 0:
                        code_list[i] = found
                        code_set.remove(c_remove)
                        code_set.add(found)
                        cover_count[balls_np[found]] += 1
                        improved = True
                        break
                    else:
                        # Restore cover_count
                        cover_count[ball_rm] += 1

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds - use more time for local search
        num_restarts = 15
        for i in range(num_restarts):
            if time.time() > deadline - 30:  # Reserve 30s for local search
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

        # Local search on the best code found - use more time
        if best_code is not None and time.time() < deadline - 10:
            ls_code = local_search(best_code, 42, deadline - 5)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Final refinement pass
        if best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case (shouldn't be called for this task, but included for completeness)
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

    # Generic greedy
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
