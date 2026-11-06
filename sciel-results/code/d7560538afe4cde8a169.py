"""Improved: multi-start greedy + strong local search + simulated annealing on conflict graph."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))
    n = len(nums)

    # Precompute conflict bitmasks
    adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            s = ai + bj
            p = ai * bj
            if p % s == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    degrees = [bin(adj[i]).count('1') for i in range(n)]

    def greedy_mis(order):
        """Greedy independent set given vertex order."""
        selected = 0
        for v in order:
            if not (adj[v] & selected):
                selected |= (1 << v)
        return selected

    def mask_to_size(mask):
        return bin(mask).count('1')

    # Strong local search: try all 1-for-2 and 2-for-3 swaps
    def local_search(mask, max_iter=200):
        cur = mask
        cur_size = mask_to_size(cur)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < DEADLINE - 3:
            improved = False
            it += 1
            # 1-for-2: remove one selected, try to add two that become available
            for i in range(n):
                if not (cur & (1 << i)):
                    continue
                trial = cur & ~(1 << i)
                added = 0
                for j in range(n):
                    if trial & (1 << j):
                        continue
                    if adj[j] & trial:
                        continue
                    if adj[i] & (1 << j):
                        trial |= (1 << j)
                        added += 1
                        if added == 2:
                            break
                if added == 2:
                    cur = trial
                    cur_size += 1
                    improved = True
                    break
            # 2-for-3: remove two selected, try to add three
            if not improved and it < max_iter // 2:
                for i in range(n):
                    if not (cur & (1 << i)):
                        continue
                    for j in range(i + 1, n):
                        if not (cur & (1 << j)):
                            continue
                        trial = cur & ~(1 << i) & ~(1 << j)
                        added = 0
                        for k in range(n):
                            if trial & (1 << k):
                                continue
                            if adj[k] & trial:
                                continue
                            if (adj[i] & (1 << k)) or (adj[j] & (1 << k)):
                                trial |= (1 << k)
                                added += 1
                                if added == 3:
                                    break
                        if added == 3:
                            cur = trial
                            cur_size += 1
                            improved = True
                            break
                    if improved:
                        break
        return cur, cur_size

    # Simulated annealing refinement
    def sa_refine(mask, max_iters=5000):
        cur = mask
        cur_size = mask_to_size(cur)
        best = cur
        best_size = cur_size
        temp = 2.0
        cooling = 0.9995
        rng = random.Random(42)
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            # Pick a random move: add, remove, or swap
            move_type = rng.random()
            if move_type < 0.4:
                # Try adding a random unselected vertex
                unselected = [i for i in range(n) if not (cur & (1 << i))]
                if not unselected:
                    continue
                v = rng.choice(unselected)
                if adj[v] & cur:
                    continue  # would create conflict
                trial = cur | (1 << v)
                delta = 1
                if delta >= 0 or rng.random() < pow(2.718281828, delta / temp):
                    cur = trial
                    cur_size += 1
                    if cur_size > best_size:
                        best = cur
                        best_size = cur_size
            elif move_type < 0.7:
                # Try removing a random selected vertex
                selected = [i for i in range(n) if cur & (1 << i)]
                if not selected:
                    continue
                v = rng.choice(selected)
                trial = cur & ~(1 << v)
                delta = -1
                if delta >= 0 or rng.random() < pow(2.718281828, delta / temp):
                    cur = trial
                    cur_size -= 1
            else:
                # Swap: remove one, add one
                selected = [i for i in range(n) if cur & (1 << i)]
                unselected = [i for i in range(n) if not (cur & (1 << i))]
                if not selected or not unselected:
                    continue
                v_rem = rng.choice(selected)
                v_add = rng.choice(unselected)
                trial = cur & ~(1 << v_rem)
                if adj[v_add] & trial:
                    continue
                trial |= (1 << v_add)
                delta = 0  # size unchanged, but structure changes
                if rng.random() < 0.3 or rng.random() < pow(2.718281828, delta / temp):
                    cur = trial
            temp *= cooling
        return best, best_size

    # Multi-start with diverse orderings
    rng_main = random.Random(123)
    best_mask = 0
    best_size = 0

    # Base orderings
    orderings = [
        sorted(range(n), key=lambda i: (degrees[i], i)),
        sorted(range(n), key=lambda i: (degrees[i], -i)),
        sorted(range(n), key=lambda i: i),
        sorted(range(n), key=lambda i: -i),
        sorted(range(n), key=lambda i: degrees[i] / (i + 1)),
        sorted(range(n), key=lambda i: -degrees[i]),
    ]

    for order in orderings:
        if time.time() > DEADLINE - 15:
            break
        mask = greedy_mis(order)
        sz = mask_to_size(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Local search on best
    best_mask, best_size = local_search(best_mask, max_iter=100)

    # Random restarts with perturbation
    for restart in range(30):
        if time.time() > DEADLINE - 10:
            break
        # Perturb best solution
        mask = best_mask
        # Remove 1-3 random selected vertices
        selected_list = [i for i in range(n) if mask & (1 << i)]
        for _ in range(min(3, len(selected_list))):
            if not selected_list:
                break
            rem = rng_main.choice(selected_list)
            mask &= ~(1 << rem)
            selected_list.remove(rem)
        # Greedily fill back
        for i in range(n):
            if not (mask & (1 << i)) and not (adj[i] & mask):
                if rng_main.random() < 0.6:
                    mask |= (1 << i)
        # Local search
        mask, sz = local_search(mask, max_iter=50)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # SA refinement on the best
    if time.time() < DEADLINE - 5:
        best_mask, best_size = sa_refine(best_mask, max_iters=3000)
        # Final local search
        best_mask, best_size = local_search(best_mask, max_iter=50)

    # Convert mask to list of numbers
    result = [nums[i] for i in range(n) if best_mask & (1 << i)]

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to odds
        return [x for x in range(1, N + 1) if x % 2 == 1]

    return result
# EVOLVE-BLOCK-END
