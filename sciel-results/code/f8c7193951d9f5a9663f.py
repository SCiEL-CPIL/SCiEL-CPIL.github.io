"""Max-weight 3-AP-free subset of [1,256] via greedy + local search."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 256
    
    # Precompute AP triples and conflicts
    pairs_for = [set() for _ in range(N)]
    for b in range(1, 257):
        for a in range(1, b):
            c = 2 * b - a
            if c > b and c <= 256:
                ai, bi, ci = a-1, b-1, c-1
                pairs_for[ai].add((bi, ci))
                pairs_for[bi].add((ai, ci))
                pairs_for[ci].add((ai, bi))
    
    def can_add(x_idx, included_mask):
        for (a, b) in pairs_for[x_idx]:
            if (included_mask >> a) & 1 and (included_mask >> b) & 1:
                return False
        return True
    
    def compute_weight(mask):
        return sum(1.0 / (i + 1) for i in range(N) if (mask >> i) & 1)
    
    def greedy_init(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << x)
        return mask
    
    best_mask = 0
    best_weight = 0.0
    
    def update_best(mask):
        nonlocal best_mask, best_weight
        w = compute_weight(mask)
        if w > best_weight:
            best_weight = w
            best_mask = mask
    
    # Initial greedy passes
    update_best(greedy_init(list(range(N))))
    update_best(greedy_init(list(range(N-1, -1, -1))))
    
    # Weighted order: prioritize high-weight elements
    update_best(greedy_init(sorted(range(N), key=lambda i: -1.0/(i+1))))
    
    # Local search with 1-swap and 1-move
    def local_search(mask, max_iter=1000):
        current_weight = compute_weight(mask)
        for _ in range(max_iter):
            if time.time() > DEADLINE - 3:
                break
            improved = False
            # Try 1-swap: remove one, add one
            for ri in range(N):
                if not (mask >> ri) & 1:
                    continue
                if time.time() > DEADLINE - 3:
                    return mask
                new_mask = mask & ~(1 << ri)
                rw = 1.0 / (ri + 1)
                for ai in range(N):
                    if (new_mask >> ai) & 1:
                        continue
                    if not can_add(ai, new_mask):
                        continue
                    aw = 1.0 / (ai + 1)
                    if aw > rw + 1e-12:
                        candidate = new_mask | (1 << ai)
                        cw = current_weight - rw + aw
                        if cw > current_weight + 1e-12:
                            mask = candidate
                            current_weight = cw
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_mask = mask
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return mask
    
    best_mask = local_search(best_mask)
    
    # Multiple random restarts
    for trial in range(100):
        if time.time() > DEADLINE - 3:
            break
        order = list(range(N))
        random.shuffle(order)
        m = greedy_init(order)
        m = local_search(m, max_iter=200)
    
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return result

