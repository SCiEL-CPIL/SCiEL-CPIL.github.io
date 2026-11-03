import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # Safety margin vs 120s wall-clock kill


def _all_words(q, n):
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)


def _build_balls(q, n, radius):
    """Build balls as list of sorted arrays of indices."""
    total = q ** n
    base = _all_words(q, n)
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    idx_of = (base * powers).sum(axis=1)
    
    if radius == 1:
        # Ball size is 1 + n*(q-1)
        ball_size = 1 + n * (q - 1)
        balls = np.zeros((total, ball_size), dtype=np.int64)
        balls[:, 0] = idx_of
        k = 1
        for pos in range(n):
            for delta in range(1, q):
                flipped = base.copy()
                flipped[:, pos] = (base[:, pos] + delta) % q
                balls[:, k] = (flipped * powers).sum(axis=1)
                k += 1
        return balls, base, powers, idx_of
    else:
        # Generic BFS for small spaces
        balls_list = []
        for i in range(total):
            seen = {i}
            frontier = [i]
            for _ in range(radius):
                nxt = []
                for widx in frontier:
                    w = base[widx]
                    for pos in range(n):
                        for v in range(q):
                            if v == w[pos]:
                                continue
                            nw = w.copy()
                            nw[pos] = v
                            j = int((nw * powers).sum())
                            if j not in seen:
                                seen.add(j)
                                nxt.append(j)
                frontier = nxt
            balls_list.append(np.array(sorted(seen), dtype=np.int64))
        maxb = max(len(b) for b in balls_list)
        balls = np.zeros((total, maxb), dtype=np.int64)
        for i, b in enumerate(balls_list):
            balls[i, :len(b)] = b
        return balls, base, powers, idx_of


def _greedy_cover(balls, total, budget_time, rng):
    """Greedy set cover with vectorized gain computation."""
    cov = np.zeros(total, dtype=np.int32)
    chosen = []
    in_code = np.zeros(total, dtype=bool)
    
    t0 = time.time()
    while True:
        if time.time() - t0 > budget_time:
            break
        uncovered = (cov == 0).astype(np.int32)
        if uncovered.sum() == 0:
            break
        
        # Compute gains: for each candidate, count how many uncovered words it covers
        # balls[i] is the set of words covered by codeword i
        gains = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if in_code[i]:
                continue
            gains[i] = uncovered[balls[i]].sum()
        
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        
        chosen.append(best)
        in_code[best] = True
        cov[balls[best]] += 1
    
    return chosen, in_code, cov


def _local_search(balls, total, chosen, in_code, covcnt, budget_time, rng):
    """Local search: remove redundant, swap, 2-for-1 exchange."""
    improved = True
    t0 = time.time()
    
    def try_remove(i):
        return (covcnt[balls[i]] >= 2).all()
    
    def do_remove(i):
        covcnt[balls[i]] -= 1
        in_code[i] = False
        chosen.remove(i)
    
    def do_add(j):
        covcnt[balls[j]] += 1
        in_code[j] = True
        chosen.append(j)
    
    def try_swap(i, j):
        b_i = balls[i]
        b_j = balls[j]
        in_j = np.zeros(total, dtype=bool)
        in_j[b_j] = True
        critical = b_i[~in_j[b_i]]
        if critical.size == 0:
            return True
        return (covcnt[critical] >= 2).all()
    
    while improved and time.time() - t0 < budget_time:
        improved = False
        
        # Pass A: Redundancy Removal
        for idx in range(len(chosen)):
            if time.time() - t0 > budget_time:
                break
            i = chosen[idx]
            if try_remove(i):
                do_remove(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Pass B: 1-for-1 Swap
        if time.time() - t0 < budget_time - 2.0:
            n_chosen = len(chosen)
            chosen_indices = list(range(n_chosen))
            rng.shuffle(chosen_indices)
            
            for idx in chosen_indices:
                if time.time() - t0 > budget_time:
                    break
                i = chosen[idx]
                
                # Try random candidates
                found = False
                for _ in range(30):
                    j = rng.randint(0, total - 1)
                    if in_code[j]:
                        continue
                    if try_swap(i, j):
                        do_remove(i)
                        do_add(j)
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        if improved:
            continue
        
        # Pass C: 2-for-1 Exchange
        if time.time() - t0 < budget_time - 5.0:
            n_chosen = len(chosen)
            if n_chosen < 250:
                pairs = list(itertools.combinations(range(n_chosen), 2))
                rng.shuffle(pairs)
            else:
                pairs = [(rng.randint(0, n_chosen - 1), rng.randint(0, n_chosen - 1)) for _ in range(500)]
                pairs = [(a, b) for a, b in pairs if a != b]
            
            for a, b in pairs:
                if time.time() - t0 > budget_time:
                    break
                if a >= len(chosen) or b >= len(chosen):
                    continue
                i = chosen[a]
                k = chosen[b]
                
                b_i = balls[i]
                b_k = balls[k]
                
                in_i = np.zeros(total, dtype=bool)
                in_i[b_i] = True
                in_k = np.zeros(total, dtype=bool)
                in_k[b_k] = True
                
                lost1 = in_i & in_k & (covcnt == 2)
                lost2 = (in_i ^ in_k) & (covcnt == 1)
                lost = lost1 | lost2
                
                if not lost.any():
                    do_remove(i)
                    do_remove(k)
                    improved = True
                    break
                
                if len(np.where(lost)[0]) > 40:
                    continue
                
                found_j = -1
                for j in range(total):
                    if time.time() - t0 > budget_time:
                        break
                    if in_code[j]:
                        continue
                    b_j = balls[j]
                    in_j = np.zeros(total, dtype=bool)
                    in_j[b_j] = True
                    if in_j[lost].all():
                        found_j = j
                        break
                
                if found_j != -1:
                    do_remove(i)
                    do_remove(k)
                    do_add(found_j)
                    improved = True
                    break
    
    return chosen, in_code, covcnt


def _repair(balls, total, chosen, in_code, covcnt, budget_time):
    """Repair: add codewords to cover any uncovered words."""
    t0 = time.time()
    covered = np.zeros(total, dtype=bool)
    for i in chosen:
        covered[balls[i]] = True
    
    while time.time() - t0 < budget_time:
        uncovered = ~covered
        if not uncovered.any():
            break
        
        best = -1
        bestg = 0
        for i in range(total):
            if in_code[i]:
                continue
            ng = uncovered[balls[i]].sum()
            if ng > bestg:
                bestg = ng
                best = i
        
        if best < 0 or bestg == 0:
            break
        
        chosen.append(best)
        in_code[best] = True
        covcnt[balls[best]] += 1
        covered[balls[best]] = True
    
    return chosen, in_code, covcnt


def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    rng = random.Random(42)
    
    # Fallback for very large spaces
    if total > 5_000_000:
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tet for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(range(n))]
    
    balls, base, powers, idx_of = _build_balls(q, n, radius)
    
    def time_left():
        return _DEADLINE - (time.time() - t0)
    
    best_chosen = None
    best_len = total + 1
    
    # Phase 1: Greedy initialization with multiple random restarts
    num_restarts = 5
    restart_budget = max(2.0, time_left() / (num_restarts * 3))
    
    for restart in range(num_restarts):
        if time_left() < 10.0:
            break
        
        # Greedy with random tie-breaking
        cov = np.zeros(total, dtype=np.int32)
        chosen = []
        in_code = np.zeros(total, dtype=bool)
        
        greedy_t0 = time.time()
        greedy_budget = min(restart_budget, time_left() - 5.0)
        
        while time.time() - greedy_t0 < greedy_budget:
            uncovered = (cov == 0).astype(np.int32)
            if uncovered.sum() == 0:
                break
            
            gains = np.zeros(total, dtype=np.int32)
            for i in range(total):
                if in_code[i]:
                    continue
                gains[i] = uncovered[balls[i]].sum()
            
            # Random tie-breaking among top candidates
            max_gain = gains.max()
            if max_gain == 0:
                break
            top_candidates = np.where(gains == max_gain)[0]
            best = int(rng.choice(top_candidates))
            
            chosen.append(best)
            in_code[best] = True
            cov[balls[best]] += 1
        
        # Local search
        covcnt = np.zeros(total, dtype=np.int32)
        for i in chosen:
            covcnt[balls[i]] += 1
        
        ls_budget = min(15.0, time_left() - 5.0)
        chosen, in_code, covcnt = _local_search(balls, total, chosen, in_code, covcnt, ls_budget, rng)
        
        # Repair if needed
        chosen, in_code, covcnt = _repair(balls, total, chosen, in_code, covcnt, max(1.0, time_left() - 2.0))
        
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = list(chosen)
    
    # Phase 2: Iterated Local Search (Perturb and Repair)
    if best_chosen is not None and time_left() > 20.0:
        ils_budget = time_left() - 10.0
        ils_t0 = time.time()
        
        current_chosen = list(best_chosen)
        current_len = len(current_chosen)
        current_in_code = np.zeros(total, dtype=bool)
        for i in current_chosen:
            current_in_code[i] = True
        current_covcnt = np.zeros(total, dtype=np.int32)
        for i in current_chosen:
            current_covcnt[balls[i]] += 1
        
        while time.time() - ils_t0 < ils_budget:
            # Perturb: remove 2-3 random codewords
            perturb_size = min(3, max(2, len(current_chosen) // 50))
            if len(current_chosen) <= perturb_size:
                break
            
            to_remove = rng.sample(current_chosen, perturb_size)
            for i in to_remove:
                current_covcnt[balls[i]] -= 1
                current_in_code[i] = False
                current_chosen.remove(i)
            
            # Repair
            repair_budget = min(5.0, time_left() - 5.0)
            current_chosen, current_in_code, current_covcnt = _repair(
                balls, total, current_chosen, current_in_code, current_covcnt, repair_budget
            )
            
            # Local search
            ls_budget = min(10.0, time_left() - 5.0)
            current_chosen, current_in_code, current_covcnt = _local_search(
                balls, total, current_chosen, current_in_code, current_covcnt, ls_budget, rng
            )
            
            if len(current_chosen) < best_len:
                best_len = len(current_chosen)
                best_chosen = list(current_chosen)
    
    # Final verification and repair
    if best_chosen is not None and time_left() > 2.0:
        covered = np.zeros(total, dtype=bool)
        for i in best_chosen:
            covered[balls[i]] = True
        
        if not covered.all():
            in_code = np.zeros(total, dtype=bool)
            for i in best_chosen:
                in_code[i] = True
            covcnt = np.zeros(total, dtype=np.int32)
            for i in best_chosen:
                covcnt[balls[i]] += 1
            
            best_chosen, in_code, covcnt = _repair(balls, total, best_chosen, in_code, covcnt, max(1.0, time_left() - 0.5))
    
    if best_chosen is None:
        # Fallback: single greedy
        best_chosen, _, _ = _greedy_cover(balls, total, max(1.0, time_left()), rng)
    
    return [base[i].tolist() for i in best_chosen]
