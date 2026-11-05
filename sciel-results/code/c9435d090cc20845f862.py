import time
import itertools

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses recursive block construction with greedy ordering, followed by
    greedy deletion passes (backward, forward, random) to shorten the word.
    """
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # --- Step 1: Build a valid superpermutation via recursive construction ---
    # Standard recursive: for m in 2..n, take each (m-1)-perm P from current word
    # and form P + [m] + P, then order blocks to maximize overlap.

    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1)-permutations in order of first appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        # Build blocks: P + [m] + P
        blocks = [list(p) + [m] + list(p) for p in perms]
        num_blocks = len(blocks)
        if num_blocks == 0:
            continue

        # Greedy nearest-neighbor: start with block 0, pick next block
        # that maximizes suffix/prefix overlap.
        current_idx = 0
        new_word = blocks[current_idx][:]
        visited = [False] * num_blocks
        visited[current_idx] = True

        for _ in range(num_blocks - 1):
            best_j = -1
            best_ov = -1
            tail = new_word
            for j in range(num_blocks):
                if visited[j]:
                    continue
                b = blocks[j]
                # Maximum useful overlap is m-2
                max_ov = min(m - 2, len(tail), len(b))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if tail[-k:] == b[:k]:
                        ov = k
                        break
                if ov > best_ov:
                    best_ov = ov
                    best_j = j
            if best_j == -1:
                break
            visited[best_j] = True
            new_word.extend(blocks[best_j][best_ov:])

        word = new_word

    current_word = word[:]

    # --- Step 2: Precompute permutation lookup ---
    all_perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(all_perms)

    # Encode each permutation as an integer (base n+1)
    base = n + 1
    pow_base = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_base[i] = pow_base[i - 1] * base

    code_to_pidx = {}
    for idx, p in enumerate(all_perms):
        c = 0
        for i in range(n):
            c += p[i] * pow_base[n - 1 - i]
        code_to_pidx[c] = idx

    def window_code(w, start):
        c = 0
        for i in range(n):
            c += w[start + i] * pow_base[n - 1 - i]
        return c

    # --- Step 3: Greedy deletion passes ---
    def counts_from_word(w):
        cnt = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            wc = window_code(w, i)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                cnt[pidx] += 1
        return cnt

    def can_delete(w, counts, i):
        """Check if deleting position i keeps all permutations present."""
        L = len(w)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        if start_j > end_j:
            return True, []
        affected = []
        for j in range(start_j, end_j + 1):
            wc = window_code(w, j)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                if counts[pidx] <= 1:
                    return False, []
                affected.append(pidx)
        return True, affected

    # Strategy 1: Backward scan (delete from right to left)
    for _pass in range(50):
        if time.time() > deadline:
            break
        improved = False
        L = len(current_word)
        if L < n:
            break
        counts = counts_from_word(current_word)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            ok, affected = can_delete(current_word, counts, i)
            if ok:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break
        if not improved:
            break

    # Strategy 2: Forward scan
    for _pass in range(50):
        if time.time() > deadline:
            break
        improved = False
        L = len(current_word)
        if L < n:
            break
        counts = counts_from_word(current_word)
        for i in range(L):
            if time.time() > deadline:
                break
            ok, affected = can_delete(current_word, counts, i)
            if ok:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break
        if not improved:
            break

    # Strategy 3: Alternating backward/forward
    for _pass in range(100):
        if time.time() > deadline:
            break
        improved = False
        L = len(current_word)
        if L < n:
            break
        counts = counts_from_word(current_word)
        if _pass % 2 == 0:
            idx_range = range(L - 1, -1, -1)
        else:
            idx_range = range(L)
        for i in idx_range:
            if time.time() > deadline:
                break
            ok, affected = can_delete(current_word, counts, i)
            if ok:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break
        if not improved:
            break

    # Strategy 4: Random sampling
    import random
    for _pass in range(20):
        if time.time() > deadline:
            break
        improved = False
        L = len(current_word)
        if L < n:
            break
        counts = counts_from_word(current_word)
        positions = list(range(L))
        random.shuffle(positions)
        for i in positions:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            ok, affected = can_delete(current_word, counts, i)
            if ok:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
        if not improved:
            break

    return "".join(map(str, current_word))
