import time
import itertools

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 120.0 - 8.0   # 8s safety margin

    # Index every t-subset
    N = 0
    idx = {}
    for c in itertools.combinations(range(v), t):
        idx[c] = N
        N += 1

    # Precompute mask for every k-block
    all_blocks = []
    all_masks = []
    for b in itertools.combinations(range(v), k):
        m = 0
        for c in itertools.combinations(b, t):
            m |= (1 << idx[c])
        all_blocks.append(b)
        all_masks.append(m)

    def time_ok():
        return time.time() < deadline

    full = (1 << N) - 1
    remaining = full

    # Greedy max-coverage
    blocks = []
    used = set()
    while remaining and time_ok():
        best_i = -1
        best_cov = -1
        for i, m in enumerate(all_masks):
            if all_blocks[i] in used:
                continue
            cov = (remaining & m).bit_count()
            if cov > best_cov:
                best_cov = cov
                best_i = i
        if best_i < 0 or best_cov <= 0:
            break
        blocks.append(all_blocks[best_i])
        used.add(all_blocks[best_i])
        remaining &= ~all_masks[best_i]

    # Local search: repeatedly try to drop a redundant block or replace one
    def rebuild():
        masks = []
        union = 0
        for b in blocks:
            i = all_blocks.index(b)
            masks.append(all_masks[i])
            union |= all_masks[i]
        return masks, union

    masks, union = rebuild()
    for _ in range(300):
        if not time_ok():
            break
        improved = False
        n = len(blocks)
        for ri in range(n):
            others = 0
            for j in range(n):
                if j != ri:
                    others |= masks[j]
            gap = masks[ri] & ~others
            if gap == 0:
                blocks.pop(ri)
                masks.pop(ri)
                union = 0
                for m in masks:
                    union |= m
                improved = True
                break
            best_j = -1
            for j, m in enumerate(all_masks):
                if all_blocks[j] in blocks:
                    continue
                if (gap & ~m) == 0:
                    best_j = j
                    break
            if best_j >= 0:
                blocks[ri] = all_blocks[best_j]
                masks[ri] = all_masks[best_j]
                union = 0
                for m in masks:
                    union |= m
                improved = True
                break
        if not improved:
            break

    return blocks
