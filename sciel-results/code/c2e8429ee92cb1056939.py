import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    MAXV = 128
    N = MAXV - 1  # 127 values: 2..128

    # Precompute primes and prime-exponent vectors for each value 2..128
    # vec[i] = list of (prime, exponent) for value i+2
    primes = []
    for x in range(2, MAXV + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)

    # exponent vectors: list of tuples of (prime_index, exp)
    exp_vecs = []
    for x in range(2, MAXV + 1):
        n = x
        vec = []
        for pi, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                e = 0
                while n % p == 0:
                    n //= p
                    e += 1
                vec.append((pi, e))
        if n > 1:
            # n is a prime
            # find its index in primes
            pi = primes.index(n)
            vec.append((pi, 1))
        exp_vecs.append(tuple(vec))

    def vec_add(a, b):
        """Add two exponent vectors (tuples of (pi, exp))."""
        if not a:
            return b
        if not b:
            return a
        # merge
        di = {}
        for pi, e in a:
            di[pi] = e
        for pi, e in b:
            di[pi] = di.get(pi, 0) + e
        return tuple(sorted(di.items()))

    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)

    # --- Fast conflict count using exponent vectors ---
    def count_conflicts(seq):
        """Count number of duplicate product pairs using exponent vectors."""
        n = len(seq)
        if n < 2:
            return 0
        # Map from product-vector to count
        prod_count = {}
        for i in range(n):
            cur = ()
            for j in range(i, n):
                cur = vec_add(cur, exp_vecs[seq[j] - 2])
                prod_count[cur] = prod_count.get(cur, 0) + 1
        conflicts = 0
        for v in prod_count.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 20 * conf

    # --- Fast incremental conflict check for insertion ---
    def check_insert(seq, pos, x):
        """Check if inserting x at pos creates any new product conflicts.
        Returns True if valid (no new conflicts)."""
        n = len(seq)
        if n == 0:
            return True
        # Compute all existing products as exponent vectors
        existing = set()
        for i in range(n):
            cur = ()
            for j in range(i, n):
                cur = vec_add(cur, exp_vecs[seq[j] - 2])
                existing.add(cur)
        # New products involving x
        new_prods = set()
        # Left extensions: x * seq[pos-1] * ... * seq[i]
        p = exp_vecs[x - 2]
        new_prods.add(p)
        for i in range(pos - 1, -1, -1):
            p = vec_add(p, exp_vecs[seq[i] - 2])
            new_prods.add(p)
        # Right extensions: x * seq[pos] * ... * seq[j]
        p = exp_vecs[x - 2]
        for j in range(pos, n):
            p = vec_add(p, exp_vecs[seq[j] - 2])
            new_prods.add(p)
        # Check overlap
        if new_prods & existing:
            return False
        return True

    # --- Simulated annealing with bitmask neighborhood ---
    current_seq = initial_seq[:]
    current_score = get_score(current_seq)
    current_conflicts = count_conflicts(current_seq)

    T = 8.0
    cool_rate = 0.9995
    no_improve = 0
    step = 0

    while time.time() < deadline - 3:
        step += 1
        r = random.random()

        if r < 0.45:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if not check_insert(current_seq, pos, x):
                continue
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        elif r < 0.75:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        else:
            # Try a swap: remove one element, add another
            if len(current_seq) < 2:
                continue
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = random.choice(available)
            temp = current_seq[:idx] + current_seq[idx+1:]
            pos = 0
            for i, v in enumerate(temp):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = temp[:pos] + [x] + temp[pos:]

        new_score = get_score(new_seq)
        delta = new_score - current_score

        if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
            current_seq = new_seq
            current_score = new_score
            current_conflicts = count_conflicts(current_seq)

            if current_conflicts == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 3.0)
            else:
                no_improve += 1

        T *= cool_rate
        if T < 0.1:
            T = 0.1

        if no_improve > 1200:
            no_improve = 0
            T = max(T, 4.0)
            if current_seq:
                for _ in range(min(3, len(current_seq))):
                    if random.random() < 0.4:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq = current_seq[:idx] + current_seq[idx+1:]
                    else:
                        in_seq = set(current_seq)
                        available = [x for x in range(2, 129) if x not in in_seq]
                        if available:
                            x = random.choice(available)
                            pos = 0
                            for i, v in enumerate(current_seq):
                                if v < x:
                                    pos = i + 1
                                else:
                                    break
                            current_seq = current_seq[:pos] + [x] + current_seq[pos:]
                current_score = get_score(current_seq)
                current_conflicts = count_conflicts(current_seq)

    # Final polish passes
    for pass_num in range(5):
        in_best = set(best_seq)
        changed = True
        while changed:
            changed = False
            if pass_num == 0:
                candidates = list(range(2, 129))
            elif pass_num == 1:
                candidates = list(range(128, 1, -1))
            elif pass_num == 2:
                candidates = list(range(2, 129))
                random.shuffle(candidates)
            elif pass_num == 3:
                def prime_factor_count(x):
                    count = 0
                    d = 2
                    n = x
                    while d * d <= n:
                        while n % d == 0:
                            count += 1
                            n //= d
                        d += 1
                    if n > 1:
                        count += 1
                    return count
                candidates = [x for x in range(2, 129) if x not in in_best]
                candidates.sort(key=lambda x: (prime_factor_count(x), x))
            else:
                candidates = list(range(2, 129))
                random.shuffle(candidates)
                candidates.sort(key=lambda x: random.random() * x)

            for x in candidates:
                if x not in in_best:
                    pos = 0
                    for i, v in enumerate(best_seq):
                        if v < x:
                            pos = i + 1
                        else:
                            break
                    if check_insert(best_seq, pos, x):
                        best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                        in_best.add(x)
                        changed = True
                        best_len = len(best_seq)
                        break

    # One more pass
    in_best = set(best_seq)
    for x in range(2, 129):
        if x not in in_best:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if check_insert(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                best_len = len(best_seq)

    return best_seq
