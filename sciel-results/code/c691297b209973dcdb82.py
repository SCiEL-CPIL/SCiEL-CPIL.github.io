import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin

    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Precompute factorizations
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    # Get exponent vectors for all numbers 2..40
    all_nums = list(range(2, 41))
    exp_vectors = {}
    for n in all_nums:
        f = factorize(n)
        vec = tuple((p, f[p]) for p in sorted(f.keys()))
        exp_vectors[n] = vec

    def check_valid(selected):
        """Check if all subset products are distinct using exponent vectors."""
        n = len(selected)
        if n == 0:
            return True
        vectors = [exp_vectors[num] for num in selected]
        
        # Build exponent sum vectors for all subsets
        # Use a dict-based approach for speed
        seen = set()
        for mask in range(1 << n):
            sum_dict = {}
            for i in range(n):
                if mask & (1 << i):
                    for p, e in vectors[i]:
                        sum_dict[p] = sum_dict.get(p, 0) + e
            key = tuple(sorted(sum_dict.items())) if sum_dict else ()
            if key in seen:
                return False
            seen.add(key)
        return True

    # Step 1: Per-prime construction (baseline, gives 16)
    selected_numbers = []
    for p in primes:
        exponents = []
        exp = 1
        while p ** exp <= 40:
            exponents.append(exp)
            exp += 1
        
        best_subset = []
        for r in range(len(exponents), -1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in itertools.combinations(exponents, r):
                subset_sums = set()
                is_dissociated = True
                for mask in range(1 << len(subset)):
                    s = 0
                    for i in range(len(subset)):
                        if mask & (1 << i):
                            s += subset[i]
                    if s in subset_sums:
                        is_dissociated = False
                        break
                    subset_sums.add(s)
                if is_dissociated:
                    best_subset = list(subset)
                    found = True
                    break
            if found:
                break
        
        for e in best_subset:
            selected_numbers.append(p ** e)

    best_set = selected_numbers[:]
    if not check_valid(best_set):
        best_set = []

    # Step 2: Try to add more numbers via greedy + local search
    if time.time() < deadline - 5:
        # Try adding each candidate number
        current = best_set[:]
        for candidate in all_nums:
            if time.time() > deadline - 5:
                break
            if candidate in current:
                continue
            test = current + [candidate]
            if check_valid(test):
                current = test
        if len(current) > len(best_set):
            best_set = current[:]

        # Try swapping: remove one, add another
        if time.time() < deadline - 3:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(len(best_set)):
                    if time.time() > deadline - 3:
                        break
                    for candidate in all_nums:
                        if candidate in best_set:
                            continue
                        new_set = best_set[:i] + best_set[i+1:] + [candidate]
                        if len(set(new_set)) == len(new_set) and check_valid(new_set):
                            best_set = new_set
                            improved = True
                            break
                    if improved:
                        break

    # Final verification
    if best_set and check_valid(best_set):
        if all(2 <= x <= 40 for x in best_set) and len(set(best_set)) == len(best_set):
            return [int(x) for x in best_set]

    # Fallback
    return [int(x) for x in selected_numbers]
