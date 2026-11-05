import time
import random
import itertools


def _standard_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                           if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word


def _build_word_from_path(perms, path, n):
    """Build a superpermutation word from a given path of permutation indices."""
    w = []
    for idx in path:
        perm = perms[idx]
        block = list(perm) + [n] + list(perm)
        overlap = 0
        min_len = min(len(w), len(block))
        for o in range(min_len, -1, -1):
            if w[len(w) - o:] == block[:o]:
                overlap = o
                break
        w += block[overlap:]
    return w


def _local_search_remove(S, n, all_perms, deadline):
    """Local search: repeatedly try to remove characters that don't break coverage."""
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    best_S = S[:]
    best_len = len(S)

    def try_remove(i):
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)

        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j + n])
            if p in all_perms and counts.get(p, 0) == 1:
                return False

        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j + n])
            if p in all_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]

        S.pop(i)

        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - n)
        for k in range(start_new, end_new + 1):
            p = tuple(S[k:k + n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1

        return True

    stuck_passes = 0
    max_stuck = 15

    while time.time() < deadline:
        removals_in_pass = 0

        for order in ['forward', 'backward', 'random']:
            if time.time() > deadline:
                break

            if order == 'forward':
                indices = list(range(len(S)))
            elif order == 'backward':
                indices = list(range(len(S) - 1, -1, -1))
            else:
                indices = list(range(len(S)))
                random.shuffle(indices)

            i = 0
            while i < len(indices):
                if time.time() > deadline:
                    break
                pos = indices[i]
                if pos >= len(S):
                    i += 1
                    continue
                if try_remove(pos):
                    removals_in_pass += 1
                    best_S = S[:]
                    best_len = len(S)
                    if order == 'forward':
                        indices = list(range(pos, len(S)))
                        i = 0
                    elif order == 'backward':
                        indices = list(range(pos - 1, -1, -1))
                        i = 0
                    else:
                        indices = list(range(len(S)))
                        random.shuffle(indices)
                        i = 0
                else:
                    i += 1

        if removals_in_pass > 0:
            stuck_passes = 0
        else:
            stuck_passes += 1
            if stuck_passes >= max_stuck:
                break

    if len(S) < len(best_S):
        best_S = S[:]

    return best_S


def _two_opt_path(perms, overlap_mat, initial_path, deadline, max_iters=50):
    """2-opt improvement on permutation ordering path."""
    N = len(perms)
    path = initial_path[:]
    
    def calc_overlap(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlap_mat[p[i]][p[i + 1]]
        return total

    current_overlap = calc_overlap(path)
    best_overlap = current_overlap
    best_path = path[:]

    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # 2-opt: try reversing segments
        for i in range(N - 2):
            if time.time() > deadline:
                break
            for j in range(i + 1, N - 1):
                if j == N - 1:
                    old_ov = overlap_mat[path[i]][path[i + 1]]
                    new_ov = overlap_mat[path[i]][path[j]]
                    delta = new_ov - old_ov
                else:
                    old_ov1 = overlap_mat[path[i]][path[i + 1]]
                    old_ov2 = overlap_mat[path[j]][path[j + 1]]
                    new_ov1 = overlap_mat[path[i]][path[j]]
                    new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                    delta = (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                
                if delta > 0:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    current_overlap += delta
                    improved = True
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = path[:]
                    break
            if improved:
                break
        
        if not improved:
            # Try Or-Opt: move single element
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(N):
                    if abs(i - j) <= 1 or j == i:
                        continue
                    # Calculate delta for moving path[i] to position j
                    # This is complex; skip for now and rely on 2-opt
                    pass
            break
    
    return best_path, best_overlap


def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    all_perms = set(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)

    # Get all (n-1)-permutations from standard construction
    std_word = _standard_superperm(n)
    perms, seen = [], set()
    for i in range(len(std_word) - (n - 2)):
        window = tuple(std_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)

    N = len(perms)

    # Build overlap matrix: overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
    overlap_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            if i == j:
                overlap_mat[i][j] = n - 1
                continue
            pj = perms[j]
            ov = 0
            for k in range(n - 2, -1, -1):
                if k > 0 and pi[-k:] == pj[:k]:
                    ov = k
                    break
            overlap_mat[i][j] = ov

    # Initial path: standard order
    naive_path = list(range(N))

    # Multi-start with different initial orderings
    random.seed(42)
    best_word = None
    best_len = float('inf')

    # Strategy 1: Standard order + local search
    if time.time() < deadline - 10:
        word = _build_word_from_path(perms, naive_path, n)
        S = list(word)
        S = _local_search_remove(S, n, all_perms, deadline - 5)
        if len(S) < best_len:
            best_len = len(S)
            best_word = S[:]

    # Strategy 2: Nearest Neighbor + 2-opt + local search
    if time.time() < deadline - 30:
        # Nearest Neighbor construction
        unvisited = list(range(N))
        random.shuffle(unvisited)
        nn_path = [unvisited.pop()]
        while unvisited:
            last = nn_path[-1]
            best_idx = -1
            best_ov = -1
            for idx in unvisited:
                ov = overlap_mat[last][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            nn_path.append(best_idx)
            unvisited.remove(best_idx)

        # 2-opt improvement
        nn_path, _ = _two_opt_path(perms, overlap_mat, nn_path, deadline - 20, max_iters=30)

        word = _build_word_from_path(perms, nn_path, n)
        S = list(word)
        S = _local_search_remove(S, n, all_perms, deadline - 5)
        if len(S) < best_len:
            best_len = len(S)
            best_word = S[:]

    # Strategy 3: Multiple random restarts with 2-opt
    restarts = 3
    for r in range(restarts):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(perms)  # Shuffle to get different starting points
        # Rebuild overlap matrix for shuffled perms
        overlap_mat2 = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat2[i][j] = n - 1
                    continue
                pj = perms[j]
                ov = 0
                for k in range(n - 2, -1, -1):
                    if k > 0 and pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat2[i][j] = ov

        # Random initial path
        random_path = list(range(N))
        random.shuffle(random_path)

        # 2-opt
        random_path, _ = _two_opt_path(perms, overlap_mat2, random_path, deadline - 10, max_iters=20)

        word = _build_word_from_path(perms, random_path, n)
        S = list(word)
        S = _local_search_remove(S, n, all_perms, deadline - 5)
        if len(S) < best_len:
            best_len = len(S)
            best_word = S[:]

    # Final: ensure validity and return
    if best_word is None:
        best_word = list(_standard_superperm(n))

    # Verify all permutations are present
    found = set()
    for i in range(len(best_word) - n + 1):
        p = tuple(best_word[i:i + n])
        if p in all_perms:
            found.add(p)

    if len(found) < total_perms:
        # Fallback to standard construction
        return "".join(map(str, _standard_superperm(n)))

    return "".join(map(str, best_word))
