# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s kill

    # Build forbidden set
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)

    # Build conflict bitmask for each vertex
    # conflict_mask[r] is a bitmask where bit s is set if r and s conflict
    # r and s conflict if (r-s)%m in forbidden or (s-r)%m in forbidden
    conflict_mask = [0] * m
    for r in range(m):
        mask = 0
        for f in forbidden_list:
            s1 = (r - f) % m
            s2 = (r + f) % m
            mask |= (1 << s1)
            mask |= (1 << s2)
        mask &= ~(1 << r)  # remove self
        conflict_mask[r] = mask

    def is_compatible(candidate, selected_mask):
        """Check if candidate is compatible with all elements in selected_mask."""
        return (conflict_mask[candidate] & selected_mask) == 0

    def greedy_fill(selected_mask, order, time_check=None):
        """Greedily add elements from order to selected_mask."""
        count = bin(selected_mask).count('1')
        for cand in order:
            if time_check is not None and time_check():
                break
            if (selected_mask >> cand) & 1:
                continue
            if is_compatible(cand, selected_mask):
                selected_mask |= (1 << cand)
                count += 1
        return selected_mask, count

    def local_search(selected_mask, count, time_budget, rng):
        """Local search: try add, swap, remove operations with greedy fill."""
        start_ls = time.time()
        best_mask = selected_mask
        best_count = count

        for iteration in range(10000):
            if time.time() - start_ls > time_budget:
                break

            op = rng.random()

            if op < 0.4:
                # Try to add a random element
                cand = rng.randint(0, m - 1)
                if not ((selected_mask >> cand) & 1) and is_compatible(cand, selected_mask):
                    new_mask = selected_mask | (1 << cand)
                    new_count = count + 1
                    # Greedy fill
                    order = list(range(m))
                    rng.shuffle(order)
                    new_mask, new_count = greedy_fill(new_mask, order,
                                                       lambda: time.time() - start_ls > time_budget)
                    selected_mask, count = new_mask, new_count
                    if count > best_count:
                        best_mask, best_count = selected_mask, count

            elif op < 0.8:
                # Try swap: remove one, add one
                if count == 0:
                    continue
                # Pick a random element to remove
                elems = [i for i in range(m) if (selected_mask >> i) & 1]
                if not elems:
                    continue
                rem = rng.choice(elems)
                new_mask = selected_mask & ~(1 << rem)
                new_count = count - 1

                # Try to add a random compatible element
                cand = rng.randint(0, m - 1)
                if not ((new_mask >> cand) & 1) and is_compatible(cand, new_mask):
                    new_mask |= (1 << cand)
                    new_count += 1

                # Greedy fill
                order = list(range(m))
                rng.shuffle(order)
                new_mask, new_count = greedy_fill(new_mask, order,
                                                   lambda: time.time() - start_ls > time_budget)
                if new_count >= count:
                    selected_mask, count = new_mask, new_count
                    if count > best_count:
                        best_mask, best_count = selected_mask, count

            else:
                # Try remove one element and greedy fill
                if count == 0:
                    continue
                elems = [i for i in range(m) if (selected_mask >> i) & 1]
                if not elems:
                    continue
                rem = rng.choice(elems)
                new_mask = selected_mask & ~(1 << rem)
                new_count = count - 1

                order = list(range(m))
                rng.shuffle(order)
                new_mask, new_count = greedy_fill(new_mask, order,
                                                   lambda: time.time() - start_ls > time_budget)
                if new_count >= count:
                    selected_mask, count = new_mask, new_count
                    if count > best_count:
                        best_mask, best_count = selected_mask, count

        return best_mask, best_count

    def extract_set(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    def verify(s_list):
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i], s_list[j]
                if ((a - b) % m) in forbidden or ((b - a) % m) in forbidden:
                    return False
        return True

    best_mask = 0
    best_count = 0

    # Multi-start greedy with different orderings
    # Use CRT-inspired orderings and random orderings
    num_starts = 2000
    for seed in range(num_starts):
        if time.time() > deadline - 30:
            break

        rng = random.Random(seed)
        order = list(range(m))

        if seed % 3 == 0:
            # CRT-inspired: sort by (residue mod 5, residue mod 41)
            order.sort(key=lambda x: (x % 5, x % 41))
        elif seed % 3 == 1:
            # Reverse CRT order
            order.sort(key=lambda x: (x % 41, x % 5))
        else:
            rng.shuffle(order)

        mask, count = greedy_fill(0, order, lambda: time.time() > deadline - 30)

        if count > best_count:
            best_count = count
            best_mask = mask

    # Local search on best solution
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        rng = random.Random(999)
        improved_mask, improved_count = local_search(best_mask, best_count, remaining, rng)
        if improved_count > best_count:
            best_count = improved_count
            best_mask = improved_mask

    # Additional local search with different seeds if time permits
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        for extra_seed in range(3):
            if time.time() > deadline - 2:
                break
            rng = random.Random(1000 + extra_seed)
            order = list(range(m))
            rng.shuffle(order)
            mask, count = greedy_fill(0, order, lambda: time.time() > deadline - 2)
            if count > best_count:
                best_count = count
                best_mask = mask
            improved_mask, improved_count = local_search(best_mask, best_count, 10, rng)
            if improved_count > best_count:
                best_count = improved_count
                best_mask = improved_mask

    result = extract_set(best_mask)

    # Verify
    if not verify(result):
        # Fallback: simple sequential greedy
        mask = 0
        for i in range(m):
            if is_compatible(i, mask):
                mask |= (1 << i)
        result = extract_set(mask)
        if not verify(result):
            # Absolute fallback
            result = [0]

    return np.array(sorted(result), dtype=np.int64)


# EVOLVE-BLOCK-END
