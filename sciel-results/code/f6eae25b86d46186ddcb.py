import time
import itertools
import random

def construct(q, n, radius):
    # Handle the specific task case: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1()
    
    # General fallback: greedy set cover with local search
    total = q ** n
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Build coverage masks
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    coverage = []
    for i, w in enumerate(words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                idx = word_to_idx[tuple(nw)]
                mask |= (1 << idx)
        coverage.append(mask)
    
    start = time.time()
    deadline = start + 100.0
    
    # Greedy set cover
    uncovered = (1 << total) - 1
    code = []
    used = [False] * total
    while uncovered:
        if time.time() > deadline - 10:
            break
        best = -1
        best_cnt = 0
        for c in range(total):
            if used[c]:
                continue
            cnt = (coverage[c] & uncovered).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        if best == -1:
            break
        code.append(best)
        used[best] = True
        uncovered &= ~coverage[best]
    
    # Local search: try removals
    if code and time.time() < deadline - 5:
        cover_count = [0] * total
        for c in code:
            m = coverage[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                m = coverage[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = coverage[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    code.pop(i)
                    improved = True
                    break
    
    return [words[i] for i in code]


def _solve_ternary_7_1():
    """Solve K_3(7,1) - minimum ternary covering code of length 7, radius 1.
    
    Uses a structured approach:
    1. Greedy set cover initialization with multiple random orders
    2. Aggressive local search with removals and swaps
    3. Uses bit-parallel operations for speed
    """
    N = 3 ** 7  # 2187
    start_time = time.time()
    deadline = start_time + 105.0
    
    # Precompute all words and their indices
    words = [list(w) for w in itertools.product(range(3), repeat=7)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute ball masks (Hamming balls of radius 1)
    ball_masks = []
    for i in range(N):
        w = words[i]
        mask = 0
        mask |= (1 << i)  # center
        for pos in range(7):
            orig = w[pos]
            for nv in range(3):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                j = word_to_idx[tuple(nw)]
                mask |= (1 << j)
        ball_masks.append(mask)
    
    # Phase 1: Multiple greedy initializations with different random orders
    best_code = None
    best_size = float('inf')
    
    # Try several random greedy orderings
    attempts = 0
    while time.time() < deadline - 20 and attempts < 15:
        attempts += 1
        order = list(range(N))
        random.shuffle(order)
        
        uncovered = (1 << N) - 1
        code = []
        used = [False] * N
        
        for c in order:
            if not uncovered:
                break
            if used[c]:
                continue
            # Check if this codeword covers any uncovered word
            # If the best uncovered word is in this ball, pick it
            # But greedy picks max coverage, so we need to be smarter
            pass
        
        # Actually do proper greedy with the random order as tie-breaker
        uncovered = (1 << N) - 1
        code = []
        used = [False] * N
        order_set = set(order)
        order_pos = {c: i for i, c in enumerate(order)}
        
        while uncovered:
            best_c = -1
            best_cnt = 0
            best_pos = N
            for c in order:
                if used[c]:
                    continue
                cnt = (ball_masks[c] & uncovered).bit_count()
                pos = order_pos[c]
                if cnt > best_cnt or (cnt == best_cnt and pos < best_pos):
                    best_cnt = cnt
                    best_c = c
                    best_pos = pos
            if best_c == -1:
                break
            code.append(best_c)
            used[best_c] = True
            uncovered &= ~ball_masks[best_c]
        
        if not uncovered:
            # Local search
            code = _local_search(code, ball_masks, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
    
    if best_code is None:
        # Emergency: just do one greedy
        uncovered = (1 << N) - 1
        code = []
        used = [False] * N
        while uncovered:
            best_c = -1
            best_cnt = 0
            for c in range(N):
                if used[c]:
                    continue
                cnt = (ball_masks[c] & uncovered).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            used[best_c] = True
            uncovered &= ~ball_masks[best_c]
        best_code = code
    
    return [words[i] for i in best_code]


def _local_search(code, ball_masks, deadline):
    """Local search: removals then swaps."""
    N = 2187
    code = list(code)
    
    # Compute cover counts
    def compute_cover_counts(codewords):
        cc = [0] * N
        for c in codewords:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cc[idx] += 1
                m ^= lsb
        return cc
    
    # Phase A: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        cc = compute_cover_counts(code)
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cc[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                code.pop(i)
                improved = True
                break
    
    # Phase B: Try swaps - replace one codeword with another to enable removal
    if time.time() < deadline - 5:
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            cc = compute_cover_counts(code)
            # Find codewords that are "weakly redundant" - they only uniquely cover a few words
            # Try swapping each codeword with a random other to see if we can reduce size
            swapped = False
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                c_orig = code[i]
                # Find words uniquely covered by c_orig
                unique_words = []
                m = ball_masks[c_orig]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cc[idx] == 1:
                        unique_words.append(idx)
                    m ^= lsb
                
                if not unique_words:
                    # This is redundant, should have been caught
                    code.pop(i)
                    swapped = True
                    break
                
                # Try to find a replacement that covers unique_words and doesn't create new uncovered
                # A replacement codeword c_new must cover all unique_words
                # and after removing c_orig and adding c_new, no word becomes uncovered
                # Simplified: try a few random replacements
                for _ in range(5):
                    c_new = random.randrange(N)
                    if c_new == c_orig or c_new in code:
                        continue
                    # Check if c_new covers all unique_words
                    new_mask = ball_masks[c_new]
                    all_covered = True
                    for uw in unique_words:
                        if not (new_mask & (1 << uw)):
                            all_covered = False
                            break
                    if not all_covered:
                        continue
                    # Check: after swap, are all words still covered?
                    # Words that were only covered by c_orig and not by c_new become uncovered
                    # But we already checked unique_words are covered by c_new
                    # Words that were covered by c_orig and c_new and no other - still covered
                    # Words that were not covered by c_orig but were covered by nothing else - still uncovered
                    # Actually we need: for every word, if it was covered by c_orig and by no other codeword,
                    # it must be covered by c_new. That's the unique_words check.
                    # And c_new doesn't need to cover anything else for validity, as long as
                    # no word loses its last covering codeword.
                    # A word w loses coverage if cc[w] == 1 and c_orig was its only coverer
                    # and c_new doesn't cover w. But that's exactly unique_words.
                    # So the swap is valid!
                    code[i] = c_new
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                break
    
    # Final removal pass
    if time.time() < deadline - 2:
        cc = compute_cover_counts(code)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cc[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cc[idx] -= 1
                        m ^= lsb
                    code.pop(i)
                    changed = True
                    break
    
    return code
