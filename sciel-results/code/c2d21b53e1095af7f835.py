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

    # Initial Candidates - focus on patterns that might have cancellations
    candidates = []

    # 1. Baseline
    candidates.append([1]*12)

    # 2. Alternating signs
    candidates.append([1 if i%2==0 else -1 for i in range(12)])

    # 3. Try patterns based on known sparse square roots
    # e.g., (1+x+x^2+x^3+x^4+x^5)^2 has 11 terms
    # But we need exactly 12 terms in P
    # Try: 1+x+x^2+x^3+x^4+x^5+x^6+x^7+x^8+x^9+x^10 - x^11
    candidates.append([1]*11 + [-1])
    candidates.append([1]*11 + [1])
    candidates.append([1,1,1,1,1,1,1,1,1,1,-1,-1])
    candidates.append([1,1,1,1,1,1,1,1,1,-1,1,-1])

    # 4. Random small ints with fixed support
    random.seed(42)
    for _ in range(20):
        c = [random.randint(-5, 5) for _ in range(12)]
        for i in range(12):
            if c[i] == 0:
                c[i] = 1
        candidates.append(c)

    # 5. Try patterns with more structure
    # e.g., pairs that might cancel
    candidates.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    candidates.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    candidates.append([1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2])

    for init_c in candidates:
        if time.time() > deadline:
            break

        curr_c = np.array(init_c, dtype=np.int64)
        curr_score = evaluate(curr_c)

        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr_c.copy()

        # Enhanced Local Search with simulated annealing
        no_improvement_count = 0
        max_no_improvement = 3000
        temperature = 2.0  # Start with some temperature for exploration
        cooling_rate = 0.9995

        while time.time() < deadline and no_improvement_count < max_no_improvement:
            # Pick a random index to modify
            idx = random.randint(0, 11)

            # Pick a random new value with bias towards small values
            new_val = random.randint(-8, 8)
            if new_val == 0:
                new_val = 1 if random.random() < 0.5 else -1

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
                # Simulated annealing: accept worse solutions with some probability
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temperature, 0.1)):
                    curr_score = new_score
                    no_improvement_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr_c.copy()
                else:
                    # Revert
                    curr_c[idx] = old_val
                    no_improvement_count += 1

            temperature *= cooling_rate

        if best_score <= 12:
            break

    # If we still have time, try more aggressive search with different approaches
    if time.time() < deadline - 10:
        # Try to find better solutions by modifying the best found so far
        if best_poly is not None:
            curr_c = best_poly.copy()
            curr_score = evaluate(curr_c)

            # More intensive local search on the best solution
            no_improvement_count = 0
            max_no_improvement = 5000
            temperature = 1.0

            while time.time() < deadline - 5 and no_improvement_count < max_no_improvement:
                # Try modifying 1-2 coefficients at a time
                num_mods = random.choice([1, 1, 1, 2])
                indices = random.sample(range(12), num_mods)

                old_vals = [curr_c[i] for i in indices]
                new_vals = []
                for i in indices:
                    val = random.randint(-8, 8)
                    if val == 0:
                        val = 1 if random.random() < 0.5 else -1
                    new_vals.append(val)

                for i, val in zip(indices, new_vals):
                    curr_c[i] = val

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
                    for i, val in zip(indices, old_vals):
                        curr_c[i] = val
                    no_improvement_count += 1

                temperature *= 0.999

    # Convert best_poly to list
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    # Ensure final coefficient is nonzero (it should be by construction, but just in case)
    # If best_poly ends with 0, trim it.
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
