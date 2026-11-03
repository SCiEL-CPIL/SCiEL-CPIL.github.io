import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    def index_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        digits.reverse()
        return digits
    
    def digits_to_index(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx
    
    # Precompute ball bitsets
    ball_bitsets = [0] * total_words
    for idx in range(total_words):
        digits = index_to_digits(idx)
        mask = 1 << idx
        for pos in range(n):
            orig = digits[pos]
            for v in range(q):
                if v == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = v
                nidx = digits_to_index(new_digits)
                mask |= (1 << nidx)
        ball_bitsets[idx] = mask
    
    def is_valid(cover):
        union = 0
        for c in cover:
            union |= ball_bitsets[c]
        return union == full_mask
    
    def minimalize(cover):
        cover_set = set(cover)
        changed = True
        while changed:
            changed = False
            for c in list(cover_set):
                rest = cover_set - {c}
                union = 0
                for r in rest:
                    union |= ball_bitsets[r]
                if union == full_mask:
                    cover_set.remove(c)
                    changed = True
        return list(cover_set)
    
    def greedy_cover():
        uncovered = full_mask
        cover = []
        remaining = list(range(total_words))
        random.shuffle(remaining)
        while uncovered:
            best_idx = -1
            best_count = -1
            best_list = []
            for c in remaining:
                cnt = (uncovered & ball_bitsets[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = c
                    best_list = [c]
                elif cnt == best_count and cnt > 0:
                    best_list.append(c)
            if best_idx == -1 or best_count == 0:
                break
            # Random tie-breaking
            chosen = random.choice(best_list)
            cover.append(chosen)
            uncovered &= ~ball_bitsets[chosen]
            remaining.remove(chosen)
        return cover
    
    def local_improve(cover):
        cover_set = set(cover)
        all_set = set(range(total_words))
        improved = True
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            if not cover_set:
                break
            c = random.choice(list(cover_set))
            candidates = all_set - cover_set
            if not candidates:
                break
            c_prime = random.choice(list(candidates))
            new_set = cover_set - {c} | {c_prime}
            if is_valid(new_set):
                cover_set = new_set
                cover_set = set(minimalize(cover_set))
                improved = True
        return list(cover_set)
    
    best_cover = None
    best_size = float('inf')
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        cover = greedy_cover()
        cover = minimalize(cover)
        if is_valid(cover):
            cover = local_improve(cover)
            if is_valid(cover):
                if len(cover) < best_size:
                    best_size = len(cover)
                    best_cover = cover[:]
    
    if best_cover is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_cover = [digits_to_index(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [index_to_digits(c) for c in best_cover]
