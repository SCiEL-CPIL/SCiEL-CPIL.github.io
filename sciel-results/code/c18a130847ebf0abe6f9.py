import math
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # First 25 primes (enough for square-free search up to ~15 primes)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97]
    num_primes = len(primes)

    # Precompute: prod_from[i][k] = product of primes[i..i+k-1], or inf if not enough primes
    INF = 10**30
    max_k_needed = 16
    prod_from = [[INF] * (max_k_needed + 1) for _ in range(num_primes + 1)]
    for i in range(num_primes, -1, -1):
        prod_from[i][0] = 1
        for k in range(1, max_k_needed + 1):
            if i + k <= num_primes:
                prod_from[i][k] = prod_from[i + 1][k - 1] * primes[i]
            else:
                prod_from[i][k] = INF

    # Score computation for a square-free n given as a bitmask of selected primes
    # Uses local bit positions (0 to k-1) instead of global indices for efficiency
    # Two divisors are coprime iff their local bitmasks are disjoint.
    
    def calc_score_squarefree(selected_indices):
        """
        selected_indices: list of indices into primes[]
        Returns the tau_perp score for n = product of primes[i] for i in selected_indices.
        """
        k = len(selected_indices)
        if k == 0:
            return 0
        # Build all divisors as (value, local_bitmask)
        # local_bitmask uses bits 0..k-1 corresponding to position in selected_indices
        divs = [(1, 0)]
        for pos, idx in enumerate(selected_indices):
            p = primes[idx]
            bit = 1 << pos
            new_divs = []
            for (val, mask) in divs:
                new_divs.append((val * p, mask | bit))
            divs.extend(new_divs)
        # Sort by value
        divs.sort(key=lambda x: x[0])
        score = 0
        for i in range(len(divs) - 1):
            if divs[i][1] & divs[i + 1][1] == 0:
                score += 1
        return score

    # Global best
    best_n = 2
    best_score = 0
    best_selected = []  # list of prime indices
    score_cache = {}

    def update_best(selected_indices, n):
        nonlocal best_n, best_score, best_selected
        if n < 2 or n > 10**18:
            return
        key = tuple(sorted(selected_indices))
        if key in score_cache:
            score = score_cache[key]
        else:
            score = calc_score_squarefree(selected_indices)
            score_cache[key] = score
        if score > best_score:
            best_score = score
            best_n = n
            best_selected = selected_indices[:]

    # DFS to find best square-free subset of size k
    def dfs_subset(idx, current_prod, selected, k):
        nonlocal best_score
        if time.time() > deadline:
            return

        # If we've selected k primes, evaluate
        if len(selected) == k:
            if current_prod <= 10**18:
                update_best(selected, current_prod)
            return

        # Pruning: not enough primes left
        remaining = num_primes - idx
        needed = k - len(selected)
        if remaining < needed:
            return

        # Pruning: even the smallest possible product of `needed` more primes
        # from index idx exceeds the remaining budget
        if current_prod * prod_from[idx][needed] > 10**18:
            return

        # Option 1: include primes[idx]
        new_prod = current_prod * primes[idx]
        if new_prod <= 10**18:
            selected.append(idx)
            dfs_subset(idx + 1, new_prod, selected, k)
            selected.pop()

        # Option 2: skip primes[idx]
        # Only if we can still fill `needed` primes from idx+1 onward
        if num_primes - (idx + 1) >= needed:
            # Prune: can we still reach the needed count with product constraint?
            if current_prod * prod_from[idx + 1][needed] <= 10**18:
                dfs_subset(idx + 1, current_prod, selected, k)

    # Search for subsets of size 15, 14, 13 (descending, as larger sets likely score higher)
    for k in [15, 14, 13, 12]:
        if time.time() > deadline - 20:
            break
        selected = []
        dfs_subset(0, 1, selected, k)

    # Local refinement: try swapping one selected prime for a larger unselected one
    # to potentially allow a different combination, or try adding/removing primes
    def local_refine(selected_indices, max_iterations=200):
        nonlocal best_n, best_score, best_selected
        current = selected_indices[:]
        if not current:
            return current
        current_n = 1
        for idx in current:
            current_n *= primes[idx]
        key = tuple(sorted(current))
        current_score = score_cache.get(key, calc_score_squarefree(current))
        if key not in score_cache:
            score_cache[key] = current_score

        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1

            # Try swapping each selected prime with a larger unselected prime
            current_set = set(current)
            for i, idx in enumerate(current):
                # Try replacing primes[idx] with primes[j] for j > idx not in current
                for j in range(idx + 1, num_primes):
                    if j in current_set:
                        continue
                    new_prod = current_n // primes[idx] * primes[j]
                    if new_prod > 10**18:
                        break
                    new_selected = current[:]
                    new_selected[i] = j
                    new_key = tuple(sorted(new_selected))
                    new_score = score_cache.get(new_key, calc_score_squarefree(new_selected))
                    if new_key not in score_cache:
                        score_cache[new_key] = new_score
                    if new_score > current_score:
                        current = new_selected
                        current_n = new_prod
                        current_score = new_score
                        current_set = set(current)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_selected = current[:]
                        break  # restart scan after a swap

            # Try adding a prime (if product allows)
            for j in range(num_primes):
                if j in current_set:
                    continue
                new_prod = current_n * primes[j]
                if new_prod > 10**18:
                    continue
                new_selected = current + [j]
                new_key = tuple(sorted(new_selected))
                new_score = score_cache.get(new_key, calc_score_squarefree(new_selected))
                if new_key not in score_cache:
                    score_cache[new_key] = new_score
                if new_score > current_score:
                    current = new_selected
                    current_n = new_prod
                    current_score = new_score
                    current_set = set(current)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                        best_selected = current[:]
                    break  # restart scan after an add

            # Try removing a prime (if it improves score)
            for i in range(len(current)):
                if len(current) <= 1:
                    break
                new_selected = current[:]
                new_selected.pop(i)
                new_prod = 1
                for idx in new_selected:
                    new_prod *= primes[idx]
                new_key = tuple(sorted(new_selected))
                new_score = score_cache.get(new_key, calc_score_squarefree(new_selected))
                if new_key not in score_cache:
                    score_cache[new_key] = new_score
                if new_score > current_score:
                    current = new_selected
                    current_n = new_prod
                    current_score = new_score
                    current_set = set(current)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                        best_selected = current[:]
                    break  # restart scan after a remove

        return current

    # Refine the best found so far
    if best_score > 0 and best_selected:
        refined = local_refine(best_selected, max_iterations=300)

    # Also try a few heuristic candidates and refine them
    # Candidate: primorial of first k primes
    for k in range(10, 16):
        if time.time() > deadline - 5:
            break
        prod = 1
        valid = True
        for i in range(k):
            prod *= primes[i]
            if prod > 10**18:
                valid = False
                break
        if valid:
            sel = list(range(k))
            update_best(sel, prod)
            if time.time() < deadline - 10:
                local_refine(sel, max_iterations=50)

    # Try non-consecutive prime subsets: skip some small primes to include larger ones
    # This can sometimes yield better scores by changing divisor structure
    if time.time() < deadline - 15:
        # Heuristic: try subsets that skip prime 2 or 3
        skip_patterns = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # skip 2
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # skip 3
            [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # skip 5
        ]
        for pattern in skip_patterns:
            if time.time() > deadline - 10:
                break
            prod = 1
            valid = True
            for idx in pattern:
                if idx < num_primes:
                    prod *= primes[idx]
                    if prod > 10**18:
                        valid = False
                        break
            if valid and len(pattern) >= 10:
                sel = pattern[:]
                update_best(sel, prod)
                if time.time() < deadline - 8:
                    local_refine(sel, max_iterations=30)

    # Final refinement
    if best_score > 0 and best_selected and time.time() < deadline - 5:
        refined = local_refine(best_selected, max_iterations=500)

    return best_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")
