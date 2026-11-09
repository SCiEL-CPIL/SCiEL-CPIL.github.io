"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using vectorized numpy."""
        a = np.array(sorted(A), dtype=np.int64)
        # Sumset A+A (with equal summands allowed): all a[i]+a[j] for i<=j
        # Use broadcasting for upper triangle
        sumset = np.unique(a[:, None] + a[None, :])
        # Product set A*A (with equal factors allowed)
        prodset = np.unique(a[:, None] * a[None, :])
        return max(len(sumset), len(prodset))

    def evaluate_partial(A, changed_idx):
        """Faster evaluation when only one element changes. Still uses numpy but we can
        skip recomputing sums/products that don't involve the changed element.
        For simplicity we just call evaluate; the vectorized version is fast enough."""
        return evaluate(A)

    # --- Strategy 1: Sequential and simple structured sets ---
    candidates = []

    # Sequential
    A_seq = list(range(1, 21))
    candidates.append(A_seq)

    # Geometric progression ratio 2
    A_geo = [2**i for i in range(20)]
    candidates.append(A_geo)

    # Arithmetic progressions
    for a in range(1, 10):
        for d in range(1, 6):
            A_ap = [a + i * d for i in range(n)]
            if max(A_ap) <= max_val:
                candidates.append(A_ap)

    # --- Strategy 2: Divisor sets of highly composite numbers ---
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
            i += 1
        return sorted(divs)

    highly_composite = [720, 2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400,
                        55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
                        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
                        3603600, 4324320, 6486480, 7207200, 10810800, 14414400]

    for N in highly_composite:
        if time.time() > deadline:
            break
        divs = [d for d in get_divisors(N) if d <= max_val]
        if len(divs) < n:
            continue
        # First 20 divisors
        candidates.append(divs[:n])
        # Last 20 divisors
        candidates.append(divs[-n:])
        # Middle 20 divisors
        if len(divs) >= 40:
            mid = len(divs) // 2
            candidates.append(divs[mid-n//2:mid+n//2])
        # Mix small and large
        if len(divs) >= 40:
            candidates.append(divs[:10] + divs[-10:])

    # --- Strategy 3: Multiplicative lattice sets ---
    # Numbers of the form 2^a * 3^b * 5^c * 7^d with small exponents
    def gen_lattice(limit):
        vals = set()
        for a in range(0, 15):
            p2 = 2**a
            if p2 > limit:
                break
            for b in range(0, 12):
                p23 = p2 * (3**b)
                if p23 > limit:
                    break
                for c in range(0, 8):
                    p235 = p23 * (5**c)
                    if p235 > limit:
                        break
                    for d in range(0, 6):
                        v = p235 * (7**d)
                        if v > limit:
                            break
                        vals.add(v)
        return sorted(vals)

    lattice_vals = gen_lattice(max_val)
    if len(lattice_vals) >= n:
        # Take first n
        candidates.append(lattice_vals[:n])
        # Take a spread
        if len(lattice_vals) >= 40:
            step = len(lattice_vals) // n
            candidates.append(lattice_vals[::step][:n])

    # --- Strategy 4: Small prime factor numbers ---
    def count_divisors(m):
        cnt = 0
        i = 1
        while i * i <= m:
            if m % i == 0:
                cnt += 1
                if i != m // i:
                    cnt += 1
            i += 1
        return cnt

    # Precompute divisor counts for numbers up to 5000 (to keep it fast)
    div_counts = [0] * (max_val + 1)
    for i in range(1, int(max_val**0.5) + 1):
        for j in range(i, max_val + 1, i):
            div_counts[j] += 1
            if i != j // i:
                div_counts[j] += 1

    # Get top 200 numbers by divisor count
    top_div = sorted(range(1, max_val + 1), key=lambda x: div_counts[x], reverse=True)[:200]
    # Try first 20
    candidates.append(top_div[:n])
    # Try a subset
    if len(top_div) >= 40:
        candidates.append(top_div[:10] + top_div[10:20])

    # --- Evaluate all candidates ---
    for A in candidates:
        if time.time() > deadline:
            break
        if len(set(A)) != n:
            continue
        if max(A) > max_val or min(A) < 1:
            continue
        A_sorted = sorted(A)
        score = evaluate(A_sorted)
        if score < best_score:
            best_score = score
            best_set = A_sorted[:]

    # --- Local search from best and other good starts ---
    random.seed(42)

    # Build list of starting points
    starts = [best_set]
    for A in candidates:
        if len(set(A)) == n and max(A) <= max_val and min(A) >= 1:
            A_s = sorted(A)
            if A_s not in starts:
                starts.append(A_s)
    # Limit starts
    starts = starts[:10]

    for start in starts:
        if time.time() > deadline:
            break

        current = start[:]
        current_score = evaluate(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]

        # Local search with structured moves
        for iteration in range(20000):
            if time.time() > deadline:
                break

            move_type = random.random()
            improved = False

            if move_type < 0.3:
                # Replace one element with random value
                i = random.randint(0, n - 1)
                new_val = random.randint(1, max_val)
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
                    current.sort()

            elif move_type < 0.5:
                # Replace with a value near an existing element
                i = random.randint(0, n - 1)
                base = current[i]
                delta = random.randint(-5, 5)
                new_val = base + delta
                if new_val < 1 or new_val > max_val or new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
                    current.sort()

            elif move_type < 0.7:
                # Replace with a multiple of an existing element
                i = random.randint(0, n - 1)
                factor = random.choice([2, 3, 4, 5, 6, 7])
                new_val = current[i] * factor
                if new_val > max_val or new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
                    current.sort()

            elif move_type < 0.85:
                # Replace with a divisor of an existing element
                i = random.randint(0, n - 1)
                val = current[i]
                divs = []
                for d in range(2, int(val**0.5) + 1):
                    if val % d == 0:
                        divs.append(d)
                        if d != val // d:
                            divs.append(val // d)
                if not divs:
                    continue
                new_val = random.choice(divs)
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
                    current.sort()

            else:
                # Swap two elements with nearby values
                i, j = random.sample(range(n), 2)
                # Try to find a value for position i that's different
                new_val = random.randint(1, max_val)
                if new_val in current:
                    continue
                old_i = current[i]
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_i
                    current.sort()

            if improved:
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]

    # --- Simulated annealing style: allow some worsening moves ---
    if time.time() < deadline - 5:
        current = best_set[:]
        current_score = best_score
        temp = 10.0
        for iteration in range(15000):
            if time.time() > deadline - 2:
                break
            temp *= 0.999

            i = random.randint(0, n - 1)
            new_val = random.randint(1, max_val)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            diff = new_score - current_score
            if diff < 0 or random.random() < np.exp(-diff / max(temp, 0.1)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()

    return best_set
# EVOLVE-BLOCK-END
