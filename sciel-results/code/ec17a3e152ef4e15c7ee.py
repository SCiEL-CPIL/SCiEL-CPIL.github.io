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

    # 6. Additional structured candidates inspired by cancellation patterns
    # Patterns where coefficients sum to zero in certain combinations
    candidates.append([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
    candidates.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    candidates.append([2, 1, -1, -2, 1, -1, 2, -1, 1, -2, 1, -1])
    candidates.append([1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1])
    candidates.append([1, -2, 1, 2, -1, -2, 1, -2, 1, 2, -1, 2])

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

    # Constraint satisfaction approach: try to find polynomials where
    # specific pairs of convolution terms cancel exactly
    if time.time() < deadline - 5:
        # For support {0..11}, the square has terms at degrees 0..22
        # Term at degree k = sum_{i+j=k} a_i * a_j
        # We want many of these to be zero.
        # Let's try systematic search over coefficient patterns
        
        # Key insight: if a_i = a_j and a_k = -a_l with i+j = k+l, 
        # we get cancellation. Let's exploit this.
        
        # Try patterns with symmetric structure
        for pattern_type in range(10):
            if time.time() > deadline - 5:
                break
            
            if pattern_type == 0:
                # a_i = a_{11-i} (symmetric)
                base = [random.randint(-4, 4) for _ in range(6)]
                c = base + base[::-1]
            elif pattern_type == 1:
                # a_i = -a_{11-i} (anti-symmetric)
                base = [random.randint(-4, 4) for _ in range(6)]
                c = base + [-x for x in base[::-1]]
            elif pattern_type == 2:
                # Periodic pattern with period 3
                base = [random.randint(-3, 3) for _ in range(3)]
                c = base * 4
            elif pattern_type == 3:
                # Periodic pattern with period 4
                base = [random.randint(-3, 3) for _ in range(4)]
                c = base * 3
            elif pattern_type == 4:
                # Geometric-like: a_i = r^i for small r
                r = random.choice([-2, -1, 1, 2])
                c = [r**i for i in range(12)]
            elif pattern_type == 5:
                # Sparse within dense: many zeros replaced by small values
                c = [1]*12
                for _ in range(4):
                    idx = random.randint(0, 11)
                    c[idx] = random.choice([-2, -1, 1, 2])
            elif pattern_type == 6:
                # Two clusters of non-zero values
                c = [0]*12
                for i in range(0, 6):
                    c[i] = random.randint(-3, 3)
                    if c[i] == 0:
                        c[i] = 1
                for i in range(6, 12):
                    c[i] = random.randint(-3, 3)
                    if c[i] == 0:
                        c[i] = 1
            elif pattern_type == 7:
                # Fibonacci-like
                c = [1, 1]
                for i in range(10):
                    c.append((c[-1] + c[-2]) % 5 - 2)
                    if c[-1] == 0:
                        c[-1] = 1
            elif pattern_type == 8:
                # Random with constraint that sum is 0
                c = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    if c[i] == 0:
                        c[i] = 1
                # Adjust to make sum close to 0
                s = sum(c)
                idx = random.randint(0, 11)
                c[idx] -= s
                if c[idx] == 0:
                    c[idx] = 1
                if abs(c[idx]) > 64:
                    c[idx] = 1 if s > 0 else -1
            else:
                # Random with pairwise cancellation structure
                c = [0]*12
                for pair in [(0,11), (1,10), (2,9), (3,8), (4,7), (5,6)]:
                    v = random.randint(-4, 4)
                    if v == 0:
                        v = 1
                    c[pair[0]] = v
                    c[pair[1]] = -v if random.random() < 0.5 else v
            
            # Ensure exactly 12 non-zero terms
            for i in range(12):
                if c[i] == 0:
                    c[i] = 1
            # Ensure within bounds
            c = np.array([max(-64, min(64, x)) for x in c], dtype=np.int64)
            
            curr_score = evaluate(c)
            if curr_score < best_score:
                best_score = curr_score
                best_poly = c.copy()
            
            # Quick local search from this start
            no_imp = 0
            temp = 1.5
            while time.time() < deadline - 5 and no_imp < 1000:
                idx = random.randint(0, 11)
                new_val = random.randint(-6, 6)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
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
                        if best_score <= 12:
                            break
                else:
                    delta = new_score - curr_score
                    if random.random() < np.exp(-delta / max(temp, 0.1)):
                        curr_score = new_score
                        no_imp = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = c.copy()
                    else:
                        c[idx] = old_val
                        no_imp += 1
                temp *= 0.999
                if best_score <= 12:
                    break
            
            if best_score <= 12:
                break

    # Convert best_poly to list
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    # Ensure final coefficient is nonzero (it should be by construction, but just in case)
    # If best_poly ends with 0, trim it.
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
