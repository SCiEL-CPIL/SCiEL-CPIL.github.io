import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        # coeffs is a numpy array
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        # Check constraints
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    # Strategy:
    # 1. The current best program (score 1.278) uses fixed support {0..11} and local search.
    # 2. The inspiration (score 0.885) uses random supports.
    # 3. Theory: To minimize non-zero terms in the square, we need collisions in the exponents.
    #    With support {0..11}, the sums i+j range from 0 to 22. There are many collisions.
    #    With sparse support, collisions are rare.
    #    So fixed support {0..11} is likely superior.
    # 4. However, we can also try "almost" contiguous supports or different coefficient patterns.
    # 5. We will refine the local search to be more effective.
    #    - Use a better initial set of candidates.
    #    - Use a more aggressive local search with multiple restarts if time permits.
    #    - Try to drive the score below 23 (current best is 23/12 ~ 1.916? No, score is 1.2778 which is 23/18?
    #      Wait, the score is "raw value (lower is better)".
    #      The task says: "Minimize nonzero terms in the EXACT integer square."
    #      So the value to minimize is the count of non-zero terms in the square.
    #      The score reported is 1.2778.
    #      If the value is 12, score is 12.
    #      If the value is 15, score is 15.
    #      Wait, the prompt says: "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    #      Seed program: [1]*12. Square has 23 non-zero terms.
    #      Seed value = 23.
    #      Seed score = 23/23 = 1.0.
    #      Current program score = 1.2778.
    #      This implies Current Value = 23 / 1.2778 ≈ 18.
    #      So the current program found a polynomial whose square has 18 non-zero terms.
    #      We want to minimize the count of non-zero terms in the square.
    #      Target: Get as low as possible. Theoretical minimum for 12 terms?
    #      If P^2 has k non-zero terms, P must have at least sqrt(k) terms? No.
    #      If P is a sum of 12 monomials, P^2 has up to 23 terms (if all sums distinct).
    #      If we can make many sums collide and cancel, we can reduce the count.
    #      Can we get 12? That would mean P^2 is a sum of 12 monomials.
    #      This implies P is a "sparse square root".
    #      Let's aim for 18 or lower.

    # Initial Candidates
    candidates = []

    # 1. Baseline
    candidates.append([1]*12)

    # 2. Alternating
    candidates.append([1 if i%2==0 else -1 for i in range(12)])

    # 3. Random small ints with fixed support
    random.seed(42)
    for _ in range(10):
        c = [random.randint(-5, 5) for _ in range(12)]
        for i in range(12):
            if c[i] == 0:
                c[i] = 1
        candidates.append(c)

    # 4. Try to construct specific patterns that might cancel more
    # e.g. (1+x+x^2)(1+x^3+x^6)...
    # Let's just rely on the local search from good starts.

    for init_c in candidates:
        if time.time() > deadline:
            break

        curr_c = np.array(init_c, dtype=np.int64)
        curr_score = evaluate(curr_c)

        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr_c.copy()

        # Local Search
        no_improvement_count = 0
        max_no_improvement = 5000  # Increased limit
        step = 0

        while time.time() < deadline and no_improvement_count < max_no_improvement:
            step += 1

            # Pick a random index to modify
            idx = random.randint(0, 11)

            # Pick a random new value
            # Bias towards small values to increase chance of cancellation
            # Values in [-10, 10] are good.
            new_val = random.randint(-10, 10)
            if new_val == 0:
                new_val = 1 # Must be non-zero

            if curr_c[idx] == new_val:
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement_count = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr_c.copy()
                    if best_score <= 12:
                        break
            else:
                # Revert
                curr_c[idx] = old_val
                no_improvement_count += 1

        if best_score <= 12:
            break

    # If we still have time, try a few random restarts with different supports?
    # Given the time, fixed support seems best.
    # But let's try a few random supports just in case.
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random support
            support = random.sample(range(41), 12)
            support.sort()
            c = np.zeros(41, dtype=np.int64)
            for s in support:
                c[s] = random.randint(-5, 5)
                if c[s] == 0:
                    c[s] = 1
            # Trim to last nonzero
            last_nz = 40
            for i in range(40, -1, -1):
                if c[i] != 0:
                    last_nz = i
                    break
            c = c[:last_nz+1]

            curr_score = evaluate(c)
            if curr_score < best_score:
                best_score = curr_score
                best_poly = c.copy()

            # Quick local search on this support
            no_imp = 0
            while time.time() < deadline and no_imp < 1000:
                idx = random.choice(support) # Modify only existing support? Or allow moving?
                # Moving support is complex. Let's just modify values.
                new_val = random.randint(-10, 10)
                if new_val == 0:
                    new_val = 1
                if c[idx] == new_val:
                    continue
                old_val = c[idx]
                c[idx] = new_val
                new_score = evaluate(c)
                if new_score < curr_score:
                    curr_score = new_score
                    no_imp = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = c.copy()
                else:
                    c[idx] = old_val
                    no_imp += 1

    # Convert best_poly to list
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    # Ensure final coefficient is nonzero (it should be by construction, but just in case)
    # If best_poly ends with 0, trim it.
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
