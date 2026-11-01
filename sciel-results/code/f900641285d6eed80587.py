import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    balls_arr = np.zeros((N, ball_size), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls.append(neighbors)
        balls_arr[w] = neighbors
    
    # Precompute ball membership as a boolean matrix for fast operations
    # ball_membership[i, j] = True if j is in ball(i)
    ball_membership = np.zeros((N, N), dtype=bool)
    for i in range(N):
        ball_membership[i, balls_arr[i]] = True
    
    # Precompute intersection counts: for each pair (u,v), how many balls overlap
    # Actually, we need: for a given u, the set of words in ball(u) \ ball(v)
    # Let's precompute the "risk" sets more efficiently using the membership matrix
    
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def prune(selected, cover_count):
        """Remove codewords that are not essential (all their ball members have cover_count > 1)."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                ball = balls_arr[c]
                if np.min(cover_count[ball]) > 1:
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
        return selected
    
    def greedy_construct(cover_count, uncovered):
        """Greedy set cover: repeatedly pick the word covering most uncovered positions."""
        selected = []
        while np.any(uncovered):
            # Use the membership matrix for fast gain computation
            gains = ball_membership[:, uncovered].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            ball = balls_arr[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
        return selected
    
    def local_search_swap(selected, cover_count, deadline):
        """Try random swaps to enable further pruning."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            if not selected:
                break
            # Try multiple random swaps
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(len(selected))
                u = selected[i]
                v = np.random.randint(N)
                if v == u:
                    continue
                # Check safety: words in B(u) \ B(v) must have cover_count >= 2
                # risk = ball(u) & ~ball(v)
                risk_mask = ball_membership[u] & ~ball_membership[v]
                if np.any(risk_mask):
                    risk_words = np.where(risk_mask)[0]
                    if np.min(cover_count[risk_words]) < 2:
                        continue
                # Safe swap
                ball_u = balls_arr[u]
                ball_v = balls_arr[v]
                cover_count[ball_u] -= 1
                cover_count[ball_v] += 1
                selected[i] = v
                # Prune
                new_len_before = len(selected)
                selected = prune(selected, cover_count)
                if len(selected) < new_len_before:
                    improved = True
                    break
                else:
                    # Revert
                    cover_count[ball_v] -= 1
                    cover_count[ball_u] += 1
                    selected[i] = u
        return selected
    
    def local_search_2for1(selected, cover_count, deadline):
        """Try to replace two codewords with one."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            K = len(selected)
            if K < 2:
                break
            # Try random pairs
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                c1 = selected[i1]
                c2 = selected[i2]
                ball1 = balls_arr[c1]
                ball2 = balls_arr[c2]
                unique1 = ball1[cover_count[ball1] == 1]
                unique2 = ball2[cover_count[ball2] == 1]
                all_unique = np.unique(np.concatenate([unique1, unique2]))
                if len(all_unique) == 0:
                    # No unique words, so we can just remove one
                    min1 = np.min(cover_count[ball1])
                    min2 = np.min(cover_count[ball2])
                    if min1 >= min2:
                        cover_count[ball1] -= 1
                        selected.pop(i1)
                    else:
                        cover_count[ball2] -= 1
                        selected.pop(i2)
                    selected = prune(selected, cover_count)
                    improved = True
                    break
                # Find v that covers all_unique: v must be in intersection of balls of all_unique
                # v is valid if ball_membership[all_unique, v].all()
                # This is a column-wise check on the membership matrix
                valid_v_mask = ball_membership[all_unique, :].all(axis=0)
                valid_v_candidates = np.where(valid_v_mask)[0]
                if len(valid_v_candidates) > 0:
                    # Try each candidate
                    found = False
                    for v in valid_v_candidates:
                        if time.time() > deadline - 5.0:
                            break
                        ball_v = balls_arr[v]
                        # Check safety: words in (ball1 ∪ ball2) \ ball_v must have cover_count >= 2
                        # after removing c1 and c2
                        combined_mask = ball_membership[c1] | ball_membership[c2]
                        risk_mask = combined_mask & ~ball_membership[v]
                        if np.any(risk_mask):
                            risk_words = np.where(risk_mask)[0]
                            # Simulate the cover count change
                            cc_sim = cover_count.copy()
                            cc_sim[ball1] -= 1
                            cc_sim[ball2] -= 1
                            if np.min(cc_sim[risk_words]) < 2:
                                continue
                        # Perform 2-for-1
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[ball_v] += 1
                        selected.pop(max(i1, i2))
                        selected.pop(min(i1, i2))
                        selected.append(v)
                        selected = prune(selected, cover_count)
                        improved = True
                        found = True
                        break
                    if found:
                        break
        return selected
    
    def tetracode_init():
        """Initialize using tetracode product structure."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_set = []
        for tc in tetracode:
            for tail in range(3 ** 4):
                # Convert tail to 4-digit ternary
                digits = []
                t = tail
                for _ in range(4):
                    digits.append(t % 3)
                    t //= 3
                digits = digits[::-1]
                word = tc + digits
                # Convert to index
                idx = 0
                for d in word:
                    idx = idx * 3 + d
                init_set.append(idx)
        return init_set
    
    best_size = N
    best_set = None
    
    # Main search loop
    while time.time() < deadline - 10.0:
        # Try different initialization strategies
        if random.random() < 0.5:
            # Start from tetracode-based initialization
            selected = tetracode_init()
            cover_count = np.zeros(N, dtype=np.int32)
            for c in selected:
                cover_count[balls_arr[c]] += 1
            # Check if it's a valid cover
            if not np.all(cover_count > 0):
                # Fill in uncovered words with greedy
                uncovered = cover_count == 0
                selected = greedy_construct(cover_count, uncovered)
                if np.any(uncovered):
                    continue
        else:
            # Pure greedy construction
            cover_count = np.zeros(N, dtype=np.int32)
            uncovered = np.ones(N, dtype=bool)
            selected = greedy_construct(cover_count, uncovered)
            if np.any(uncovered):
                continue
        
        # 2. Initial Pruning
        selected = prune(selected, cover_count)
        
        # 3. Local Search: Swaps
        selected = local_search_swap(selected, cover_count, deadline)
        
        # 4. Local Search: 2-for-1
        selected = local_search_2for1(selected, cover_count, deadline)
        
        # 5. Final Pruning
        selected = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we're getting close to the deadline, do a final polish
        if time.time() > deadline - 20.0 and best_set is not None:
            # Try a few more targeted improvements on the best set
            cover_count_best = np.zeros(N, dtype=np.int32)
            for c in best_set:
                cover_count_best[balls_arr[c]] += 1
            best_set = prune(best_set, cover_count_best)
            best_set = local_search_swap(best_set, cover_count_best, deadline)
            best_set = local_search_2for1(best_set, cover_count_best, deadline)
            best_set = prune(best_set, cover_count_best)
            if len(best_set) < best_size:
                best_size = len(best_set)
                best_set = best_set.copy()
            break
    
    if best_set is None:
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
