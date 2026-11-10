"""Combined program: greedy baseline + local-search refinement to maximize length."""
import time
import random

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

    def compute_products(seq):
        """Return the set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

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

    def get_neighbor(seq, rng):
        """Generate a random neighbor by insertion, deletion, or swap."""
        n = len(seq)
        if n == 0:
            return seq[:]
        op = rng.choice(['insert', 'delete', 'swap'])
        if op == 'insert':
            # Try inserting a random number not in seq
            candidates = [x for x in range(2, 129) if x not in seq]
            if not candidates:
                return seq[:]
            x = rng.choice(candidates)
            pos = rng.randint(0, n)
            new_seq = seq[:pos] + [x] + seq[pos:]
            # Check strictly increasing
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
        elif op == 'delete':
            if n <= 1:
                return seq[:]
            pos = rng.randint(0, n - 1)
            return seq[:pos] + seq[pos + 1:]
        else:  # swap
            if n < 2:
                return seq[:]
            i, j = rng.sample(range(n), 2)
            new_seq = seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            # Check strictly increasing
            for k in range(len(new_seq) - 1):
                if new_seq[k] >= new_seq[k + 1]:
                    return seq[:]
            return new_seq

    # Strategy 1: Greedy baseline
    best = greedy_init()
    best_len = len(best)

    # Strategy 2: Local search to try to improve
    rng = random.Random(42)
    current = best[:]
    current_len = best_len
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        neighbor = get_neighbor(current, rng)
        if len(neighbor) < current_len:
            # Allow shorter neighbors occasionally for exploration
            if rng.random() > 0.1:
                no_improve += 1
                continue
        if is_valid(neighbor):
            if len(neighbor) > current_len:
                current = neighbor
                current_len = len(neighbor)
                no_improve = 0
                if current_len > best_len:
                    best = current[:]
                    best_len = current_len
            else:
                # Accept same-length or shorter with some probability
                if rng.random() < 0.3:
                    current = neighbor
                    no_improve += 1
                else:
                    no_improve += 1
        else:
            no_improve += 1

    # Strategy 3: Try a few different random seeds for local search
    for seed in [123, 456, 789]:
        if time.time() > deadline - 10:
            break
        rng2 = random.Random(seed)
        cur = best[:]
        cur_len = len(cur)
        no_imp = 0
        max_iter = 3000
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_imp > 3000:
                break
            nb = get_neighbor(cur, rng2)
            if is_valid(nb):
                if len(nb) > cur_len:
                    cur = nb
                    cur_len = len(nb)
                    no_imp = 0
                    if cur_len > best_len:
                        best = cur[:]
                        best_len = cur_len
                else:
                    no_imp += 1
            else:
                no_imp += 1

    # Final verification
    if not is_valid(best):
        best = greedy_init()

    return best

# EVOLVE-BLOCK-END
