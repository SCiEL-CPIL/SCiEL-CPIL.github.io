import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute prime factorizations for numbers 2..128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def factorize(n):
        """Return a tuple of exponents (one per prime) for n."""
        exps = [0] * num_primes
        for i, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                exps[i] += 1
                n //= p
        if n > 1:
            exps[prime_index[n]] += 1
        return tuple(exps)

    # Precompute factor vectors
    factor_cache = {}
    for i in range(2, 129):
        factor_cache[i] = factorize(i)

    # Precompute total exponent budget: factorization of 128!
    total_budget = [0] * num_primes
    for i in range(2, 129):
        fv = factor_cache[i]
        for j in range(num_primes):
            total_budget[j] += fv[j]
    total_budget = tuple(total_budget)

    def add_vec(a, b):
        return tuple(a[i] + b[i] for i in range(num_primes))

    def sub_vec(a, b):
        return tuple(a[i] - b[i] for i in range(num_primes))

    def check_feasible(used_budget):
        """Check if remaining budget is non-negative (feasible to extend)."""
        for i in range(num_primes):
            if total_budget[i] - used_budget[i] < 0:
                return False
        return True

    # Greedy baseline using factor vectors
    def run_greedy():
        values = []
        product_set = set()
        used_budget = [0] * num_primes
        for x in range(2, 129):
            if time.time() > deadline:
                break
            # Compute tails as factor vectors
            tails = []
            vec = factor_cache[x]
            tails.append(vec)
            for a in reversed(values):
                vec = add_vec(vec, factor_cache[a])
                tails.append(vec)
            valid = True
            for t in tails:
                if t in product_set:
                    valid = False
                    break
            if valid:
                values.append(x)
                product_set.update(tails)
                for j in range(num_primes):
                    used_budget[j] += factor_cache[x][j]
        return values, product_set, tuple(used_budget)

    best_seq, best_prods, best_budget = run_greedy()
    best_len = len(best_seq)

    # Beam search with factor vectors and feasibility pruning
    # State: (seq_tuple, product_set, used_budget_tuple)
    beam = {(): (set(), tuple([0] * num_primes))}

    def flexibility(seq, prods, used_budget, start_val):
        """Count how many numbers >= start_val can be appended without collision."""
        count = 0
        for x in range(start_val, 129):
            tails = []
            vec = factor_cache[x]
            tails.append(vec)
            for a in reversed(seq):
                vec = add_vec(vec, factor_cache[a])
                tails.append(vec)
            if not any(t in prods for t in tails):
                count += 1
        return count

    for x in range(2, 129):
        if time.time() > deadline:
            break

        next_beam = {}

        for seq, (prods, used_budget) in beam.items():
            # Option 1: Skip x
            if seq not in next_beam:
                next_beam[seq] = (prods, used_budget)

            # Option 2: Include x
            if seq and seq[-1] >= x:
                continue

            # Compute tails
            tails = []
            vec = factor_cache[x]
            tails.append(vec)
            for a in reversed(seq):
                vec = add_vec(vec, factor_cache[a])
                tails.append(vec)

            # Check collision
            has_collision = any(t in prods for t in tails)
            if has_collision:
                continue

            # Check feasibility: new used budget
            new_used = [used_budget[i] + factor_cache[x][i] for i in range(num_primes)]
            if any(total_budget[i] - new_used[i] < 0 for i in range(num_primes)):
                continue

            new_seq = seq + (x,)
            new_prods = set(prods)
            new_prods.update(tails)
            new_budget = tuple(new_used)

            next_beam[new_seq] = (new_prods, new_budget)

        # Prune beam: sort by (length desc, flexibility desc, sum asc)
        if len(next_beam) > 500:
            # For large beams, use a cheaper scoring
            scored = []
            for seq, (prods, ub) in next_beam.items():
                l = len(seq)
                # Quick flexibility estimate: count available numbers
                avail = 128 - (seq[-1] if seq else 1)
                scored.append((l, avail, -sum(seq), seq))
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            keep = scored[:400]
            beam = {s: next_beam[s] for _, _, _, s in keep}
        else:
            scored = []
            for seq, (prods, ub) in next_beam.items():
                l = len(seq)
                s_val = sum(seq) if seq else 0
                scored.append((l, -s_val, seq))
            scored.sort(key=lambda t: (-t[0], t[1]))
            keep = scored[:400]
            beam = {s: next_beam[s] for _, _, s in keep}

        # Update best
        for seq in beam:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)
                best_prods, best_budget = beam[seq]

        if time.time() > deadline:
            break

    # Local repair: try to extend or improve the best sequence
    # Try inserting skipped numbers into gaps
    if time.time() < deadline - 5:
        seq = list(best_seq)
        prods = set(best_prods)
        used = set(seq)

        # Try to insert each skipped number at each position
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            skipped = [x for x in range(2, 129) if x not in used]
            for x in skipped:
                for pos in range(len(seq) + 1):
                    # Check if x fits at pos (strictly increasing)
                    if pos > 0 and seq[pos - 1] >= x:
                        continue
                    if pos < len(seq) and seq[pos] <= x:
                        continue

                    # Try inserting x at pos
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    # Recompute all products
                    new_prods = set()
                    valid = True
                    n = len(new_seq)
                    for i in range(n):
                        prod = 1
                        for j in range(i, n):
                            prod *= new_seq[j]
                            if prod in new_prods:
                                valid = False
                                break
                            new_prods.add(prod)
                        if not valid:
                            break

                    if valid and len(new_seq) > len(seq):
                        seq = new_seq
                        prods = new_prods
                        used = set(seq)
                        best_seq = seq[:]
                        best_len = len(seq)
                        best_prods = prods
                        improved = True
                        break
                if improved:
                    break

    return best_seq


if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(res)
