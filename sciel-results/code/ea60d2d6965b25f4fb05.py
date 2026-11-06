import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    n_max = 4096
    max_sum = 3 * n_max

    def build_sums_array(S):
        sums = np.zeros(max_sum + 1, dtype=bool)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums[S[i] + S[j] + S[l]] = True
        return sums

    def can_add(x, S, sums):
        """Fast incremental conflict check for adding x to set S."""
        if sums[3 * x]:
            return False
        k = len(S)
        for s in S:
            if sums[2 * x + s] or sums[x + 2 * s]:
                return False
        for i in range(k):
            s = S[i]
            for j in range(i + 1, k):
                t = S[j]
                if sums[x + s + t]:
                    return False
        return True

    def add_x(x, S, sums):
        S.append(x)
        sums[3 * x] = True
        k = len(S)
        for i in range(k - 1):
            s = S[i]
            sums[2 * x + s] = True
            sums[x + 2 * s] = True
        for i in range(k - 1):
            s = S[i]
            for j in range(i + 1, k - 1):
                t = S[j]
                sums[x + s + t] = True

    def greedy_forward(start_list, deadline):
        S = list(start_list)
        sums = build_sums_array(S)
        existing = set(S)
        for x in range(1, n_max + 1):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        return S

    def greedy_backward(start_list, deadline):
        S = list(start_list)
        sums = build_sums_array(S)
        existing = set(S)
        for x in range(n_max, 0, -1):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        return S

    def greedy_random(start_list, order, deadline):
        S = list(start_list)
        sums = build_sums_array(S)
        existing = set(S)
        for x in order:
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        return S

    def density_swap(S, deadline, max_rounds=30):
        """Try to increase density: remove a large element, add a smaller one.
        Uses incremental sums tracking to avoid full rebuilds."""
        S = sorted(S)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            k = len(S)
            if k == 0:
                break
            # Build sums once for current S
            sums_cur = build_sums_array(S)
            existing = set(S)
            # Try removing each element (larger elements first)
            for idx in range(k - 1, -1, -1):
                if time.time() > deadline:
                    break
                orig = S[idx]
                S_no = S[:idx] + S[idx + 1:]
                sums_no = build_sums_array(S_no)
                existing_no = set(S_no)
                # Find the smallest element we can add
                best = None
                for new_x in range(1, min(n_max + 1, orig + 1)):
                    if new_x in existing_no:
                        continue
                    if can_add(new_x, S_no, sums_no):
                        best = new_x
                        break
                if best is not None and best < orig:
                    S[idx] = best
                    S.sort()
                    return True
        return False

    def local_search(S, deadline, max_rounds=40):
        """Iterative density swaps to pack the set denser."""
        S = sorted(S)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            if not density_swap(S, deadline, max_rounds=1):
                break
        return S

    def remove_and_refill(S, deadline, target_size=None):
        """Remove some elements and try to refill with smaller ones to increase count.
        Strategy: remove the largest element, then greedily add the smallest possible
        new elements until we can't add more or hit deadline."""
        if target_size is None:
            target_size = len(S)
        S = sorted(S)
        if len(S) < 2:
            return S
        
        # Try removing 1-3 largest elements and refilling
        for num_remove in range(1, min(4, len(S) - 1) + 1):
            if time.time() > deadline:
                break
            S_try = S[:len(S) - num_remove]
            sums_try = build_sums_array(S_try)
            existing_try = set(S_try)
            
            # Greedily add smallest possible elements
            improved = False
            for x in range(1, n_max + 1):
                if time.time() > deadline:
                    break
                if x in existing_try:
                    continue
                if can_add(x, S_try, sums_try):
                    add_x(x, S_try, sums_try)
                    existing_try.add(x)
                    improved = True
                if len(S_try) >= len(S):
                    break
            
            if len(S_try) > len(S):
                return S_try
        return S

    # --- Strong seed: powers of 4 (valid B_3[1] set: 1+1+1=3 < 4, no carries) ---
    pow4 = []
    v = 1
    while v <= n_max:
        pow4.append(v)
        v *= 4
    # pow4 = [1, 4, 16, 64, 256, 1024, 4096]

    # Also try scaled powers of 4 for variety
    pow4_2 = [2 * v for v in pow4 if 2 * v <= n_max]  # [2, 8, 32, 128, 512, 2048]
    pow4_3 = [3 * v for v in pow4 if 3 * v <= n_max]  # [3, 12, 48, 192, 768, 3072]

    # --- Multiple strategies, keep the best (largest) set ---
    best = []
    best_len = 0

    def consider(S):
        nonlocal best, best_len
        if S and len(S) > best_len:
            best = sorted(S)
            best_len = len(S)

    # 1. Forward greedy from powers of 4
    consider(greedy_forward(pow4, deadline))
    # 2. Backward greedy from powers of 4
    consider(greedy_backward(pow4, deadline))
    # 3. Forward greedy from empty
    consider(greedy_forward([], deadline))
    # 4. Backward greedy from empty
    consider(greedy_backward([], deadline))
    # 5. Forward greedy from scaled powers of 4
    consider(greedy_forward(pow4_2, deadline))
    consider(greedy_backward(pow4_2, deadline))
    consider(greedy_forward(pow4_3, deadline))
    consider(greedy_backward(pow4_3, deadline))

    # 6. Local search (density swaps) on the best so far
    if time.time() < deadline - 3:
        consider(local_search(best, deadline))
    
    # 6b. Remove-and-refill on the best so far
    if time.time() < deadline - 3:
        consider(remove_and_refill(best, deadline))

    # 7. Randomized restarts with varied skeletons
    random.seed(12345)
    for attempt in range(50):
        if time.time() > deadline - 3:
            break
        # Mix: powers of 4 plus a few random small elements
        skel = list(pow4)
        extra = random.randint(0, 3)
        for _ in range(extra):
            cand = random.randint(1, 50)
            if cand not in skel:
                skel.append(cand)
        skel.sort()
        # Verify skeleton validity
        valid = True
        sk_sums = np.zeros(max_sum + 1, dtype=bool)
        k = len(skel)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s_val = skel[i] + skel[j] + skel[l]
                    if sk_sums[s_val]:
                        valid = False
                        break
                    sk_sums[s_val] = True
            if not valid:
                break
        if not valid:
            continue
        # Random order for remaining candidates
        remaining = [x for x in range(1, n_max + 1) if x not in set(skel)]
        random.shuffle(remaining)
        direction = random.choice([greedy_random, greedy_forward, greedy_backward])
        if direction == greedy_random:
            res = greedy_random(skel, remaining, deadline)
        elif direction == greedy_forward:
            res = greedy_forward(skel, deadline)
        else:
            res = greedy_backward(skel, deadline)
        consider(res)
        # Polish with local search if promising
        if len(res) >= best_len - 1 and time.time() < deadline - 3:
            consider(local_search(res, deadline))
        # Also try remove-and-refill
        if len(res) >= best_len - 1 and time.time() < deadline - 3:
            consider(remove_and_refill(res, deadline))

    # 8. Additional greedy passes with different starting points
    if time.time() < deadline - 5:
        for trial in range(15):
            if time.time() > deadline - 3:
                break
            # Start with a few small random elements
            small_start = []
            for _ in range(random.randint(1, 4)):
                cand = random.randint(1, 30)
                if cand not in small_start:
                    small_start.append(cand)
            small_start.sort()
            # Verify validity
            valid = True
            sk_sums = np.zeros(max_sum + 1, dtype=bool)
            k = len(small_start)
            for i in range(k):
                for j in range(i, k):
                    for l in range(j, k):
                        s_val = small_start[i] + small_start[j] + small_start[l]
                        if sk_sums[s_val]:
                            valid = False
                            break
                        sk_sums[s_val] = True
                if not valid:
                    break
            if not valid:
                continue
            direction = random.choice([greedy_forward, greedy_backward])
            res = direction(small_start, deadline)
            consider(res)
            if len(res) >= best_len - 1 and time.time() < deadline - 3:
                consider(local_search(res, deadline))
            if len(res) >= best_len - 1 and time.time() < deadline - 3:
                consider(remove_and_refill(res, deadline))

    # Final safety: ensure we return a valid set
    if not best:
        best = list(range(1, 10))  # fallback

    return sorted(best)[:80]
