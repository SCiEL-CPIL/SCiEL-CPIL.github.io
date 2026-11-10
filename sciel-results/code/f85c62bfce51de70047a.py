"""Combined program: greedy baseline + local search with 1-out/2-in moves."""
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

    def get_neighbor_1out2in(seq, rng):
        """1-out/2-in move: remove one element, insert up to two new elements.
        This escapes local optima where single insertions are blocked."""
        n = len(seq)
        if n == 0:
            return seq[:]

        # Choose element to remove
        rm_pos = rng.randint(0, n - 1)

        # Choose up to 2 candidates to insert (from numbers not in seq)
        in_seq = set(seq)
        candidates = [x for x in range(2, 129) if x not in in_seq]
        if not candidates:
            return seq[:]

        # Decide how many to insert: 1 or 2
        k = rng.choice([1, 1, 2, 2, 2])  # bias toward 2 for exploration
        if k > len(candidates):
            k = len(candidates)
        new_elems = rng.sample(candidates, k)

        # Remove the element
        base = seq[:rm_pos] + seq[rm_pos + 1:]
        base_n = len(base)

        # Try inserting new_elems at random positions, maintaining increasing order
        # We'll insert them and then check validity; if not increasing, skip
        # Strategy: pick positions for each new element
        positions = sorted(rng.sample(range(base_n + 1), k) if k <= base_n + 1
                           else list(range(base_n + 1))[:k])

        # Build new sequence by inserting at positions
        new_seq = []
        ei = 0
        bi = 0
        for pos in range(base_n + 1):
            while ei < k and positions[ei] == pos:
                new_seq.append(new_elems[ei])
                ei += 1
            if bi < base_n:
                new_seq.append(base[bi])
                bi += 1

        # Verify strictly increasing
        for i in range(len(new_seq) - 1):
            if new_seq[i] >= new_seq[i + 1]:
                return seq[:]  # reject, return original
        return new_seq

    def get_neighbor_insert(seq, rng):
        """Single insertion move."""
        n = len(seq)
        in_seq = set(seq)
        candidates = [x for x in range(2, 129) if x not in in_seq]
        if not candidates:
            return seq[:]
        x = rng.choice(candidates)
        pos = rng.randint(0, n)
        new_seq = seq[:pos] + [x] + seq[pos:]
        for i in range(len(new_seq) - 1):
            if new_seq[i] >= new_seq[i + 1]:
                return seq[:]
        return new_seq

    def get_neighbor_delete(seq, rng):
        """Single deletion move."""
        n = len(seq)
        if n <= 1:
            return seq[:]
        pos = rng.randint(0, n - 1)
        return seq[:pos] + seq[pos + 1:]

    # Strategy 1: Greedy baseline
    best = greedy_init()
    best_len = len(best)

    # Strategy 2: Local search with mixed moves (1-out/2-in, insert, delete)
    rng = random.Random(42)
    current = best[:]
    current_len = best_len
    no_improve = 0
    max_no_improve = 20000

    while time.time() < deadline - 8 and no_improve < max_no_improve:
        # Choose move type: bias toward 1-out/2-in
        move = rng.random()
        if move < 0.6:
            neighbor = get_neighbor_1out2in(current, rng)
        elif move < 0.85:
            neighbor = get_neighbor_insert(current, rng)
        else:
            neighbor = get_neighbor_delete(current, rng)

        if is_valid(neighbor):
            if len(neighbor) > current_len:
                current = neighbor
                current_len = len(neighbor)
                no_improve = 0
                if current_len > best_len:
                    best = current[:]
                    best_len = current_len
            else:
                # Accept same/shorter with some probability for exploration
                if rng.random() < 0.25:
                    current = neighbor
                no_improve += 1
        else:
            no_improve += 1

    # Strategy 3: More 1-out/2-in focused search with different seeds
    for seed in [7, 123, 456, 789, 1000]:
        if time.time() > deadline - 12:
            break
        rng2 = random.Random(seed)
        cur = best[:]
        cur_len = len(cur)
        no_imp = 0
        for _ in range(8000):
            if time.time() > deadline - 8:
                break
            if no_imp > 4000:
                break
            # Mostly 1-out/2-in
            if rng2.random() < 0.7:
                nb = get_neighbor_1out2in(cur, rng2)
            else:
                nb = get_neighbor_insert(cur, rng2)
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
