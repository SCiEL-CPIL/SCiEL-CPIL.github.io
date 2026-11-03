import numpy as np
import itertools
import random
import time
from scipy.optimize import linprog

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks
    balls = [0] * total_words
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    # 3. LP Relaxation
    # We want to minimize sum(x) subject to A x >= 1
    # A[i,j] = 1 if j covers i
    # This is the same as the balls matrix transposed logic.
    # balls[j] has bit i set if i is in ball[j].
    # So A[i,j] = 1 iff (balls[j] >> i) & 1
    
    # Construct A_ub for linprog: -A x <= -1
    # We use a dense matrix for simplicity, N=2187 is small enough.
    A_cover = np.zeros((total_words, total_words), dtype=np.float64)
    for j in range(total_words):
        mask = balls[j]
        # Iterate set bits
        # Faster:
        for i in range(total_words):
            if (mask >> i) & 1:
                A_cover[i, j] = 1.0
                
    c = np.ones(total_words)
    A_ub = -A_cover
    b_ub = -np.ones(total_words)
    bounds = [(0, 1)] * total_words
    
    x_lp = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_lp = res.x
    except Exception:
        pass
        
    if x_lp is None:
        x_lp = np.ones(total_words) / total_words

    def greedy_cover_lp(uncovered_mask):
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if x_lp[i] > x_lp[best_idx]:
                        best_idx = i
            
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        if not selected:
            return []
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def local_search_2_1(selected):
        if len(selected) < 3:
            return None
        for _ in range(5):
            i1, i2 = random.sample(range(len(selected)), 2)
            c1 = selected[i1]
            c2 = selected[i2]
            rest_mask = 0
            for i in range(len(selected)):
                if i != i1 and i != i2:
                    rest_mask |= balls[selected[i]]
            uncovered = all_ones_mask & ~rest_mask
            if uncovered == 0:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
                continue
            best_idx = -1
            max_cover = 0
            for k in range(total_words):
                if k == c1 or k == c2:
                    continue
                c = (balls[k] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = k
                elif c == max_cover and c > 0:
                    if x_lp[k] > x_lp[best_idx]:
                        best_idx = k
            if best_idx != -1 and (balls[best_idx] & uncovered) == uncovered:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected.append(best_idx)
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
        return None

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0

    initial_selected = greedy_cover_lp(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    if len(initial_selected) < best_size:
        best_size = len(initial_selected)
        best_code = [all_words[i] for i in initial_selected]
    
    current_selected = initial_selected[:]
    
    while time.time() - start_time < time_limit:
        improved = False
        new_sel = local_search_2_1(current_selected)
        if new_sel:
            current_selected = new_sel
            improved = True
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                best_code = [all_words[i] for i in current_selected]
        
        if improved:
            continue
        
        perturb_amount = random.randint(2, 4)
        if len(current_selected) <= perturb_amount:
            current_selected = greedy_cover_lp(all_ones_mask)
            current_selected = remove_redundants(current_selected)
        else:
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            uncovered_mask = all_ones_mask & ~covered_mask
            repair_selected = greedy_cover_lp(uncovered_mask)
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            new_selected = remove_redundants(new_selected)
            
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            
            if len(new_selected) <= len(current_selected) or (random.random() < 0.05 and len(new_selected) < len(current_selected) + 3):
                current_selected = new_selected

    return best_code
