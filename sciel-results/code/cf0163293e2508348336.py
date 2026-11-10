"""Combined: greedy baseline + fast pre-check + targeted insertion + SA with 1-out/2-in."""
import time
import random
from math import log

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before 90s limit

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_init():
        """Standard greedy: iterate 2..128, add if no tail-product collision."""
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def greedy_randomized(rng, seed_val=2):
        """Randomized greedy: try values in random order, keep valid ones."""
        values = []
        products = set()
        candidates = list(range(seed_val, 129))
        rng.shuffle(candidates)
        for x in candidates:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    def fast_precheck(seq):
        """Cheap necessary condition: no product equals 2^k.
        Any product with a prime factor >2 is automatically fine, so only
        powers of two (2,4,8,...) can collide with a singleton. O(n)."""
        seen = set()
        for x in seq:
            # x is in [2,128], so x is a power of two only if x & (x-1) == 0
            if x & (x - 1) == 0:
                if x in seen:
                    return False
                seen.add(x)
        return True

    def local_search_insertions(seq, max_iterations=5000):
        """Try to improve by inserting missing values at valid positions."""
        seq = seq[:]
        in_seq = set(seq)
        missing = [x for x in range(2, 129) if x not in in_seq]
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
            random.shuffle(missing)
            for x in missing:
                if time.time() > deadline - 10:
                    break
                n = len(seq)
                for pos in range(n + 1):
                    if pos > 0 and seq[pos-1] >= x:
                        continue
                    if pos < n and x >= seq[pos]:
                        continue
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    if not fast_precheck(new_seq):
                        continue
                    if is_valid(new_seq):
                        seq = new_seq
                        in_seq.add(x)
                        missing.remove(x)
                        improved = True
                        break
        return seq

    def get_neighbor_sa(seq, rng, temp):
        """Generate a neighbor using SA-style moves, with 1-out/2-in as a
        first-class move (the one that escaped greedy local optima)."""
        n = len(seq)
        if n == 0:
            return seq[:]

        move_type = rng.random()
        if move_type < 0.45:
            # 1-out/1-in
            rm_pos = rng.randint(0, n - 1)
            in_set = set(seq)
            candidates = [x for x in range(2, 129) if x not in in_set]
            if not candidates:
                return seq[:]
            x = rng.choice(candidates)
            base = seq[:rm_pos] + seq[rm_pos + 1:]
            base_n = len(base)
            pos = rng.randint(0, base_n)
            new_seq = base[:pos] + [x] + base[pos:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
        elif move_type < 0.85:
            # 1-out/2-in: remove one, insert two new elements at increasing positions
            rm_pos = rng.randint(0, n - 1)
            in_set = set(seq)
            candidates = [x for x in range(2, 129) if x not in in_set]
            if len(candidates) < 2:
                if not candidates:
                    return seq[:]
                x = rng.choice(candidates)
                base = seq[:rm_pos] + seq[rm_pos + 1:]
                base_n = len(base)
                pos = rng.randint(0, base_n)
                new_seq = base[:pos] + [x] + base[pos:]
                for i in range(len(new_seq) - 1):
                    if new_seq[i] >= new_seq[i + 1]:
                        return seq[:]
                return new_seq
            x1, x2 = rng.sample(candidates, 2)
            if x1 > x2:
                x1, x2 = x2, x1
            base = seq[:rm_pos] + seq[rm_pos + 1:]
            base_n = len(base)
            # pick two distinct positions in the base (0..base_n), keep sorted
            pos1 = rng.randint(0, base_n)
            pos2 = rng.randint(0, base_n)
            if pos1 == pos2:
                pos2 = (pos1 + 1) % (base_n + 1)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            # insert x1 at pos1, x2 at pos2 (shifted by 1 after first insert)
            new_seq = base[:pos1] + [x1] + base[pos1:pos2] + [x2] + base[pos2:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
        else:
            # Swap two elements (if it maintains increasing order)
            if n < 2:
                return seq[:]
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            new_seq = seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            for k in range(len(new_seq) - 1):
                if new_seq[k] >= new_seq[k + 1]:
                    return seq[:]
            return new_seq

    # Strategy 1: Multiple greedy initializations
    best = greedy_init()
    best_len = len(best)

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 30:
            break
        rng_g = random.Random(seed)
        cand = greedy_randomized(rng_g)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # Strategy 2: Targeted insertion search
    improved_seq = local_search_insertions(best[:], max_iterations=200)
    if len(improved_seq) > best_len:
        best = improved_seq
        best_len = len(best)

    # Strategy 3: Simulated annealing with 1-out/2-in moves
    rng_sa = random.Random(42)
    cur = best[:]
    cur_len = len(cur)
    best_sa = cur[:]
    best_sa_len = cur_len
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    no_improve = 0

    while time.time() < deadline - 5 and T > T_min:
        neighbor = get_neighbor_sa(cur, rng_sa, T)
        if not fast_precheck(neighbor):
            no_improve += 1
            T *= cooling
            continue
        if is_valid(neighbor):
            delta_len = len(neighbor) - cur_len
            if delta_len >= 0:
                cur = neighbor
                cur_len = len(cur)
                if cur_len > best_sa_len:
                    best_sa = cur[:]
                    best_sa_len = cur_len
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if rng_sa.random() < (2.718281828 ** (delta_len / T)):
                    cur = neighbor
                    cur_len = len(cur)
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        if no_improve > 5000:
            cur = best_sa[:]
            cur_len = best_sa_len
            no_improve = 0
            T = max(T, 0.5)

    if best_sa_len > best_len:
        best = best_sa
        best_len = best_sa_len

    # Strategy 4: Final targeted insertion pass
    final_seq = local_search_insertions(best, max_iterations=100)
    if len(final_seq) > best_len:
        best = final_seq
        best_len = len(best)

    # Final verification
    if not is_valid(best):
        best = greedy_init()

    return best

# EVOLVE-BLOCK-END
