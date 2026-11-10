import time
import random
from fractions import Fraction

def construct():
    N = 512
    MAX_A = 511
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # Precompute prime factors for each number in 1..511
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = {}
    for a in range(1, MAX_A + 1):
        pf[a] = prime_factors(a)

    # Greedy initialization: pick in descending order of 1/(512-a), i.e. descending a
    selected = []
    selected_pfs = []
    for a in range(MAX_A, 0, -1):
        if any(pf[a].issubset(pfs) or any(p in pfs for p in pf[a]) for pfs in selected_pfs):
            continue
        # Check actual coprime with all selected
        if all(all(p not in pf[b] for p in pf[a]) for b in selected):
            selected.append(a)
            selected_pfs.append(pf[a])

    def compute_score(lst):
        s = Fraction(0, 1)
        for a in lst:
            s += Fraction(1, N - a)
        return s

    def is_coprime_with_all(a, lst):
        pa = pf[a]
        for b in lst:
            if any(p in pf[b] for p in pa):
                return False
        return True

    best = list(selected)
    best_score = compute_score(best)

    # Local search: try replacing one element with a non-selected one if it improves
    def local_search(lst):
        current_score = compute_score(lst)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, a in enumerate(lst):
                # Try removing a and adding a better one
                for a_new in range(MAX_A, 0, -1):
                    if a_new in lst:
                        continue
                    if not is_coprime_with_all(a_new, lst):
                        continue
                    # Remove a, add a_new
                    candidate = lst[:i] + lst[i+1:] + [a_new]
                    cand_score = current_score - Fraction(1, N - a) + Fraction(1, N - a_new)
                    if cand_score > current_score:
                        # Verify coprimality (should be fine since we checked)
                        lst = candidate
                        current_score = cand_score
                        improved = True
                        break
                if improved:
                    break
        return lst, current_score

    best, best_score = local_search(best)

    # Multiple restarts with randomized greedy
    for restart in range(50):
        if time.time() > deadline:
            break
        random.seed(restart)
        # Randomized greedy: shuffle candidates and pick greedily
        candidates = list(range(MAX_A, 0, -1))
        random.shuffle(candidates)
        # But we want to bias toward high-value (high a) numbers
        # Use a weighted approach: sort by 1/(512-a) with some randomness
        weights = [1.0 / (N - a) for a in candidates]
        total_w = sum(weights)
        selected_r = []
        for a in candidates:
            if is_coprime_with_all(a, selected_r):
                selected_r.append(a)
        
        # Local search from this starting point
        result, score = local_search(selected_r)
        if score > best_score:
            best = result
            best_score = score

    # Try 2-for-1 swaps: remove two, add three if possible
    if time.time() < deadline:
        for _ in range(20):  # limited iterations
            if time.time() > deadline:
                break
            lst = list(best)
            found = False
            for i in range(len(lst)):
                for j in range(i+1, len(lst)):
                    # Try removing lst[i] and lst[j]
                    remaining = [x for x in lst if x != lst[i] and x != lst[j]]
                    # Find up to 3 numbers that are coprime to remaining and each other
                    gains = []
                    for a_new in range(MAX_A, 0, -1):
                        if a_new in lst:
                            continue
                        if not is_coprime_with_all(a_new, remaining):
                            continue
                        # Check if coprime to other candidates already picked
                        if any(a_new in [g[1] for g in gains] or not all(p not in pf[g[1]] for p in pf[a_new]) for g in gains):
                            continue
                        gain = Fraction(1, N - a_new)
                        gains.append((gain, a_new))
                    # Sort by gain descending, take top up to 3
                    gains.sort(key=lambda x: x[0], reverse=True)
                    top_gains = gains[:3]
                    total_gain = sum(g[0] for g in top_gains)
                    total_loss = Fraction(1, N - lst[i]) + Fraction(1, N - lst[j])
                    if total_gain > total_loss:
                        new_lst = remaining + [g[1] for g in top_gains]
                        new_score = best_score - total_loss + total_gain
                        if new_score > best_score:
                            best = new_lst
                            best_score = new_score
                            found = True
                            break
                if found:
                    break

    return best
