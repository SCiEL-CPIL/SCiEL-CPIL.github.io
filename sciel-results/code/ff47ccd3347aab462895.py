"""Covering code K_3(6,1) via branch-and-bound set cover with local search refinement."""
import time
import itertools
import random
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as numpy array."""
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)


def _build_coverage_bitsets(q, n, radius=1):
    """Build coverage as list of Python int bitsets (one per word)."""
    total = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = [0] * total
    for i, t in enumerate(all_tuples):
        mask = 1 << i  # self
        if radius >= 1:
            for pos in range(n):
                orig = t[pos]
                for new_val in range(q):
                    if new_val != orig:
                        tl = list(t)
                        tl[pos] = new_val
                        j = idx_map[tuple(tl)]
                        mask |= (1 << j)
        neighbors[i] = mask
    return neighbors, all_tuples, idx_map


def _greedy_set_cover(neighbors, total, deadline, rng=None):
    """Greedy set cover using bitsets. Returns set of chosen indices."""
    if rng is None:
        rng = random.Random(42)
    
    full_mask = (1 << total) - 1
    uncovered = full_mask
    chosen = set()
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_cnt = 0
        for i in range(total):
            if i in chosen:
                continue
            cnt = (neighbors[i] & uncovered).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
                if best_cnt == uncovered.bit_count():
                    break
        
        if best_idx == -1 or best_cnt == 0:
            break
        
        chosen.add(best_idx)
        uncovered &= ~neighbors[best_idx]
    
    return chosen


def _branch_and_bound(neighbors, total, initial_bound, deadline, max_depth=6):
    """
    Branch-and-bound for minimum set cover.
    Tries to find a cover smaller than initial_bound.
    Uses recursive search with pruning.
    """
    full_mask = (1 << total) - 1
    best_solution = None
    best_size = initial_bound
    
    # Precompute: for each word, list of codewords that cover it (for heuristic ordering)
    # Actually, we'll use a simpler approach: at each node, pick the uncovered word
    # with fewest covering candidates (most constrained)
    
    word_coverers = [[] for _ in range(total)]
    for i in range(total):
        m = neighbors[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            word_coverers[idx].append(i)
            m ^= lsb
    
    # Shuffle order for variety
    order = list(range(total))
    random.Random(123).shuffle(order)
    
    def bnb(uncovered, chosen, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        if not uncovered:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = set(chosen)
            return
        
        # Pruning: lower bound on additional codewords needed
        # Each codeword covers at most max_cover new elements
        # Simple bound: ceil(popcount(uncovered) / max_single_cover)
        unc_count = uncovered.bit_count()
        if unc_count == 0:
            return
        
        # Better bound: find max coverage any single codeword can provide to uncovered
        max_cover = 0
        for i in range(total):
            if i in chosen:
                continue
            c = (neighbors[i] & uncovered).bit_count()
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return  # Can't cover anything more
        
        import math
        lb = math.ceil(unc_count / max_cover)
        if len(chosen) + lb >= best_size:
            return
        
        # Pick the most constrained uncovered word
        # Word with fewest available covering codewords
        best_word = -1
        best_count = total + 1
        # Sample some uncovered words for speed
        unc_words = []
        m = uncovered
        sample_limit = 50
        while m and len(unc_words) < sample_limit:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            unc_words.append(idx)
            m ^= lsb
        if not unc_words:
            return
        
        for w in unc_words:
            avail = [c for c in word_coverers[w] if c not in chosen]
            if len(avail) < best_count:
                best_count = len(avail)
                best_word = w
                if best_count <= 1:
                    break
        
        if best_word == -1 or best_count == 0:
            return
        
        # Get candidates that cover best_word, ordered by coverage of remaining uncovered
        candidates = [c for c in word_coverers[best_word] if c not in chosen]
        candidates.sort(key=lambda c: (neighbors[c] & uncovered).bit_count(), reverse=True)
        
        # Try each candidate (limit to top few for speed)
        max_branches = min(len(candidates), 8)
        for c in candidates[:max_branches]:
            new_uncovered = uncovered & ~neighbors[c]
            new_chosen = chosen | {c}
            bnb(new_uncovered, new_chosen, depth + 1)
            if time.time() > deadline:
                return
    
    bnb(full_mask, set(), 0)
    
    if best_solution is not None:
        return best_solution
    return None


def _local_search_reduce(neighbors, total, chosen, deadline):
    """
    Local search to reduce the number of codewords.
    Tries: remove one if redundant, swap one for another to enable removals.
    """
    chosen = set(chosen)
    
    # Compute cover counts
    cover_count = np.zeros(total, dtype=np.int32)
    for idx in chosen:
        m = neighbors[idx]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            cover_count[j] += 1
            m ^= lsb
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Phase 1: Remove redundant codewords
        for idx in list(chosen):
            if time.time() > deadline:
                break
            # Check if any word is only covered by this codeword
            m = neighbors[idx]
            only_covered = False
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_count[j] == 1:
                    only_covered = True
                    break
                m ^= lsb
            
            if not only_covered:
                # Safe to remove
                m = neighbors[idx]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    cover_count[j] -= 1
                    m ^= lsb
                chosen.discard(idx)
                improved = True
                break
        
        # Phase 2: Try swaps
        if not improved and time.time() < deadline:
            # Try removing each codeword and see if we can find a better replacement
            # that allows further reduction
            remove_candidates = list(chosen)
            random.Random(456).shuffle(remove_candidates)
            
            for rem in remove_candidates[:20]:
                if time.time() > deadline:
                    break
                
                # Simulate removal
                m_rem = neighbors[rem]
                # Find words that would become uncovered
                would_uncover = []
                m = m_rem
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_count[j] == 1:
                        would_uncover.append(j)
                    m ^= lsb
                
                if not would_uncover:
                    # Already handled in phase 1
                    continue
                
                # Find a codeword that covers all of would_uncover
                # and maximizes additional coverage
                best_add = -1
                best_score = -1
                
                # Sample candidates
                all_candidates = [i for i in range(total) if i not in chosen]
                if len(all_candidates) > 100:
                    random.Random(789).shuffle(all_candidates)
                    all_candidates = all_candidates[:100]
                
                for add in all_candidates:
                    m_add = neighbors[add]
                    # Check if add covers all would_uncover
                    covers_all = True
                    for w in would_uncover:
                        if not (m_add >> w) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: how many additional words does add cover that are currently uncovered
                    # or how many words does it cover in total
                    score = m_add.bit_count()
                    if score > best_score:
                        best_score = score
                        best_add = add
                
                if best_add != -1:
                    # Execute swap
                    # Remove rem
                    m = neighbors[rem]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_count[j] -= 1
                        m ^= lsb
                    chosen.discard(rem)
                    
                    # Add best_add
                    m = neighbors[best_add]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_count[j] += 1
                        m ^= lsb
                    chosen.add(best_add)
                    improved = True
                    break
    
    return chosen


def _verify_coverage(neighbors, total, chosen):
    """Verify that chosen codewords cover all words."""
    full_mask = (1 << total) - 1
    covered = 0
    for idx in chosen:
        covered |= neighbors[idx]
    return covered == full_mask


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Generic fallback
    if total_words > 5000:
        return _generic_construct(q, n, radius, deadline)
    
    return _generic_bitset_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    """Specialized solver for K_3(6,1)."""
    q, n, radius = 3, 6, 1
    total_words = 729
    
    neighbors, all_tuples, idx_map = _build_coverage_bitsets(q, n, radius)
    
    # Step 1: Get initial solution via greedy
    rng = random.Random(42)
    initial = _greedy_set_cover(neighbors, total_words, deadline, rng)
    
    if not initial:
        return [list(all_tuples[i]) for i in range(total_words)]
    
    current_best = set(initial)
    current_size = len(current_best)
    
    # Step 2: Local search to reduce
    current_best = _local_search_reduce(neighbors, total_words, current_best, deadline)
    current_size = len(current_best)
    
    # Step 3: Try B&B to find smaller solution
    if time.time() < deadline - 30 and current_size > 73:
        bnb_result = _branch_and_bound(
            neighbors, total_words, current_size, deadline - 30
        )
        if bnb_result is not None and len(bnb_result) < current_size:
            current_best = bnb_result
            current_size = len(current_best)
    
    # Step 4: More local search rounds
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            before = len(current_best)
            current_best = _local_search_reduce(neighbors, total_words, current_best, deadline)
            if len(current_best) < before:
                current_size = len(current_best)
    
    # Verify
    if not _verify_coverage(neighbors, total_words, current_best):
        # Fallback to greedy
        current_best = _greedy_set_cover(neighbors, total_words, deadline, random.Random(99))
    
    return [list(all_tuples[i]) for i in sorted(current_best)]


def _generic_construct(q, n, radius, deadline):
    """Generic greedy for larger spaces."""
    words = _all_words(q, n)
    total = len(words)
    
    # Simple greedy: pick word covering most uncovered
    uncovered = set(range(total))
    chosen = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if i in chosen:
                continue
            # Count how many uncovered words are within radius
            count = 0
            for j in uncovered:
                if np.sum(words[i] != words[j]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
        
        chosen.append(best_idx)
        # Remove covered words
        to_remove = set()
        for j in uncovered:
            if np.sum(words[best_idx] != words[j]) <= radius:
                to_remove.add(j)
        uncovered -= to_remove
    
    return [words[i].tolist() for i in chosen]


def _generic_bitset_construct(q, n, radius, deadline):
    """Bitset-based solver for small spaces."""
    total = q ** n
    neighbors, all_tuples, idx_map = _build_coverage_bitsets(q, n, radius)
    
    rng = random.Random(0)
    chosen = _greedy_set_cover(neighbors, total, deadline, rng)
    
    if not chosen:
        return [list(w) for w in all_tuples]
    
    chosen = _local_search_reduce(neighbors, total, chosen, deadline)
    
    if not _verify_coverage(neighbors, total, chosen):
        chosen = _greedy_set_cover(neighbors, total, deadline, random.Random(77))
    
    return [list(all_tuples[i]) for i in sorted(chosen)]
