"""Simulated annealing + local search for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails. Gives 81 words for q=3,n=6."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword (exact check)."""
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    words_arr = np.array(words_list, dtype=np.int8)
    code_arr = np.array(code, dtype=np.int8)
    
    diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    covered = (dists <= radius).any(axis=1)
    return covered.all()


def _remove_redundant(code, all_words_arr, radius, time_limit=5.0):
    """Remove redundant codewords from a valid code. Returns reduced code."""
    start = time.time()
    code_arr = np.array(code, dtype=np.int8)
    n_codewords = len(code)
    
    # Precompute coverage: cover[i,j] = word i is covered by codeword j
    diff = all_words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    cover = (dists <= radius)
    
    # Count how many codewords cover each word
    cover_counts = cover.sum(axis=1)
    
    # Repeatedly try to remove codewords that are redundant
    removed = set()
    changed = True
    while changed and (time.time() - start) < time_limit:
        changed = False
        # Recompute coverage with current code
        current_indices = [i for i in range(n_codewords) if i not in removed]
        if len(current_indices) == 0:
            break
        cur_arr = code_arr[current_indices]
        diff = all_words_arr[:, np.newaxis, :] != cur_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        cover = (dists <= radius)
        cover_counts = cover.sum(axis=1)
        
        # Find words covered by exactly one codeword - those codewords can't be removed
        # Try removing each codeword and check validity
        for ci in range(len(current_indices)):
            if (time.time() - start) > time_limit:
                break
            # Check if removing this codeword leaves all words covered
            remaining = cover.copy()
            remaining[:, ci] = False
            if remaining.any(axis=1).all():
                removed.add(current_indices[ci])
                changed = True
                break
    
    final_indices = [i for i in range(n_codewords) if i not in removed]
    return [list(code[i]) for i in final_indices]


def _simulated_annealing(code, all_words_arr, q, n, radius, time_limit=60.0, seed=42):
    """
    Simulated annealing to minimize code size.
    Moves: remove a codeword, add a codeword, swap a codeword.
    Objective: minimize number of codewords (with penalty for uncovered words).
    """
    rng = np.random.default_rng(seed)
    start = time.time()
    
    all_words_list = [list(w) for w in all_words_arr]
    N = len(all_words_list)  # 729
    
    # Current code as list of word indices
    word_to_idx = {}
    for i, w in enumerate(all_words_list):
        word_to_idx[tuple(w)] = i
    
    code_idx = []
    for cw in code:
        idx = word_to_idx.get(tuple(cw))
        if idx is not None:
            code_idx.append(idx)
    
    current_size = len(code_idx)
    best_size = current_size
    best_code = list(code_idx)
    
    # Precompute: for each word, its neighbors (words at distance <= radius)
    # This is expensive for 729 words but we do it once
    # Actually, let's compute on-the-fly for efficiency
    
    # Compute initial coverage
    def compute_coverage(cod_idx):
        """Return boolean array of length N: True if word i is covered."""
        if len(cod_idx) == 0:
            return np.zeros(N, dtype=bool)
        code_arr = all_words_arr[cod_idx]
        diff = all_words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        return (dists <= radius).any(axis=1)
    
    current_cover = compute_coverage(code_idx)
    uncovered_count = N - current_cover.sum()
    
    # Temperature schedule
    T0 = 5.0
    T_end = 0.01
    total_time = time_limit
    # We'll run until time runs out
    
    # Track best
    best_code = list(code_idx)
    best_size = current_size
    
    # Precompute neighbor lists for each word (words at distance exactly 1)
    # For efficiency, precompute which words are at distance <= radius from each word
    # For radius=1: a word covers itself and words at distance 1
    # Number of words at distance 1 from a word of length n over alphabet q: n*(q-1)
    # For n=6, q=3: 6*2 = 12 neighbors + self = 13 words per ball
    
    # Precompute ball membership: ball[i] = set of word indices covered by word i
    # For radius=1, this is fast to compute
    ball = []
    for i in range(N):
        w = all_words_arr[i]
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = w.copy()
                nw[pos] = val
                ni = _word_to_idx(nw.tolist(), q, n)
                covered.append(ni)
        ball.append(set(covered))
    
    # Now use ball for fast coverage updates
    # coverage_count[i] = number of codewords that cover word i
    
    def compute_coverage_counts(cod_idx):
        """Return array of coverage counts for each word."""
        counts = np.zeros(N, dtype=np.int32)
        for ci in cod_idx:
            counts[np.fromiter(ball[ci], dtype=np.int32, count=len(ball[ci]))] += 1
        return counts
    
    current_counts = compute_coverage_counts(code_idx)
    
    # SA loop
    no_improve = 0
    max_no_improve = 5000
    
    while (time.time() - start) < time_limit:
        # Decrease temperature based on elapsed time
        elapsed = time.time() - start
        progress = min(elapsed / total_time, 1.0)
        T = T0 * (T_end / T0) ** progress
        
        # Choose a random move
        move_type = rng.random()
        accepted = False
        old_counts = current_counts.copy()
        old_size = current_size
        
        if move_type < 0.4 and current_size > 1:
            # Remove a random codeword
            ri = rng.integers(0, current_size)
            removed_idx = code_idx[ri]
            new_code_idx = [x for i, x in enumerate(code_idx) if i != ri]
            new_size = current_size - 1
            # Update coverage
            new_counts = old_counts - np.fromiter(ball[removed_idx], dtype=np.int32, count=len(ball[removed_idx]))
            new_uncovered = np.sum(new_counts == 0)
            
            # Delta in objective: size change + penalty for newly uncovered
            delta = (new_size - old_size) + (new_uncovered - np.sum(old_counts == 0)) * 100
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                code_idx = new_code_idx
                current_size = new_size
                current_counts = new_counts
                accepted = True
                
        elif move_type < 0.7:
            # Add a random codeword
            if current_size < N:
                # Pick a word not already in code
                existing = set(code_idx)
                new_idx = rng.integers(0, N)
                attempts = 0
                while new_idx in existing and attempts < 50:
                    new_idx = rng.integers(0, N)
                    attempts += 1
                if new_idx not in existing:
                    new_code_idx = code_idx + [new_idx]
                    new_size = current_size + 1
                    new_counts = old_counts + np.fromiter(ball[new_idx], dtype=np.int32, count=len(ball[new_idx]))
                    new_uncovered = np.sum(new_counts == 0)
                    
                    delta = (new_size - old_size) + (new_uncovered - np.sum(old_counts == 0)) * 100
                    
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        code_idx = new_code_idx
                        current_size = new_size
                        current_counts = new_counts
                        accepted = True
        
        else:
            # Swap: remove one, add one
            if current_size > 0:
                ri = rng.integers(0, current_size)
                removed_idx = code_idx[ri]
                existing = set(code_idx)
                new_idx = rng.integers(0, N)
                attempts = 0
                while (new_idx in existing) and attempts < 50:
                    new_idx = rng.integers(0, N)
                    attempts += 1
                
                if new_idx not in existing or new_idx == removed_idx:
                    new_code_idx = [x for i, x in enumerate(code_idx) if i != ri] + [new_idx]
                    new_size = current_size
                    # Remove old ball, add new ball
                    new_counts = old_counts - np.fromiter(ball[removed_idx], dtype=np.int32, count=len(ball[removed_idx]))
                    new_counts += np.fromiter(ball[new_idx], dtype=np.int32, count=len(ball[new_idx]))
                    new_uncovered = np.sum(new_counts == 0)
                    
                    delta = (new_size - old_size) + (new_uncovered - np.sum(old_counts == 0)) * 100
                    
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        code_idx = new_code_idx
                        current_size = new_size
                        current_counts = new_counts
                        accepted = True
        
        # Track best
        if current_size < best_size:
            best_size = current_size
            best_code = list(code_idx)
            no_improve = 0
        elif accepted:
            no_improve += 1
        
        # If we've been stuck for a long time, restart from best
        if no_improve > max_no_improve:
            code_idx = list(best_code)
            current_size = best_size
            current_counts = compute_coverage_counts(code_idx)
            no_improve = 0
    
    # Final: remove redundant from best
    best_word_list = [all_words_list[i] for i in best_code]
    return best_word_list


def _local_search_polish(code, all_words_arr, q, n, radius, time_limit=10.0):
    """
    Polish a valid code by trying swaps that enable further deletions.
    """
    start = time.time()
    N = len(all_words_arr)
    all_words_list = [list(w) for w in all_words_arr]
    
    word_to_idx = {}
    for i, w in enumerate(all_words_list):
        word_to_idx[tuple(w)] = i
    
    code_idx = []
    for cw in code:
        idx = word_to_idx.get(tuple(cw))
        if idx is not None:
            code_idx.append(idx)
    
    # Precompute balls
    ball = []
    for i in range(N):
        w = all_words_arr[i]
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = w.copy()
                nw[pos] = val
                ni = _word_to_idx(nw.tolist(), q, n)
                covered.append(ni)
        ball.append(set(covered))
    
    def compute_coverage_counts(cod_idx):
        counts = np.zeros(N, dtype=np.int32)
        for ci in cod_idx:
            counts[np.fromiter(ball[ci], dtype=np.int32, count=len(ball[ci]))] += 1
        return counts
    
    def is_valid(cod_idx):
        counts = compute_coverage_counts(cod_idx)
        return np.all(counts > 0)
    
    def try_remove_redundant(cod_idx):
        """Try to remove redundant codewords, return reduced list."""
        counts = compute_coverage_counts(cod_idx)
        changed = True
        while changed:
            if (time.time() - start) > time_limit:
                break
            changed = False
            for ci in range(len(cod_idx)):
                if (time.time() - start) > time_limit:
                    break
                # Check if removing cod_idx[ci] is safe
                new_counts = counts - np.fromiter(ball[cod_idx[ci]], dtype=np.int32, count=len(ball[cod_idx[ci]]))
                if np.all(new_counts > 0):
                    cod_idx = [x for i, x in enumerate(cod_idx) if i != ci]
                    counts = new_counts
                    changed = True
                    break
        return cod_idx
    
    # Phase 1: Remove redundant
    code_idx = try_remove_redundant(code_idx)
    
    # Phase 2: Try single swaps to enable further reduction
    for iteration in range(200):
        if (time.time() - start) > time_limit:
            break
        
        best_swap = None
        best_size = len(code_idx)
        
        for ci in range(len(code_idx)):
            if (time.time() - start) > time_limit:
                break
            removed = code_idx[ci]
            existing = set(code_idx)
            
            # Try a few random candidates for replacement
            for _ in range(20):
                new_idx = int(np.random.randint(0, N))
                if new_idx in existing:
                    continue
                
                trial = [x for i, x in enumerate(code_idx) if i != ci] + [new_idx]
                # Quick check: is it valid?
                counts = compute_coverage_counts(trial)
                if not np.all(counts > 0):
                    continue
                
                # Try to reduce
                reduced = try_remove_redundant(trial)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_swap = reduced
        
        if best_swap is not None:
            code_idx = best_swap
        else:
            break
    
    # Final reduction
    code_idx = try_remove_redundant(code_idx)
    
    return [all_words_list[i] for i in code_idx]


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    all_words_arr = np.array(words_list, dtype=np.int8)
    
    # Step 1: Start from direct sum tetracode (81 words for q=3, n=6)
    initial_code = _direct_sum_tetracode(q, n, radius)
    
    # Verify it's valid
    if not _verify_code(initial_code, q, n, radius):
        # Fallback: greedy
        initial_code = words_list
    
    best_code = initial_code
    best_size = len(initial_code)
    
    # Step 2: Remove redundant codewords first
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 2:
        reduced = _remove_redundant(best_code, all_words_arr, radius, time_limit=min(remaining, 5.0))
        if _verify_code(reduced, q, n, radius) and len(reduced) < best_size:
            best_code = reduced
            best_size = len(reduced)
    
    # Step 3: Simulated annealing
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 15
    if remaining > 10:
        try:
            sa_code = _simulated_annealing(best_code, all_words_arr, q, n, radius, 
                                           time_limit=min(remaining, 80.0), seed=42)
            if _verify_code(sa_code, q, n, radius) and len(sa_code) < best_size:
                best_code = sa_code
                best_size = len(sa_code)
        except Exception:
            pass
    
    # Step 4: Local search polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 3:
        try:
            polished = _local_search_polish(best_code, all_words_arr, q, n, radius,
                                            time_limit=min(remaining, 15.0))
            if _verify_code(polished, q, n, radius) and len(polished) < best_size:
                best_code = polished
                best_size = len(polished)
        except Exception:
            pass
    
    # Step 5: Final redundancy removal
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 2
    if remaining > 1:
        try:
            final = _remove_redundant(best_code, all_words_arr, radius, time_limit=min(remaining, 3.0))
            if _verify_code(final, q, n, radius) and len(final) < best_size:
                best_code = final
                best_size = len(final)
        except Exception:
            pass
    
    return best_code

# EVOLVE-BLOCK-END
