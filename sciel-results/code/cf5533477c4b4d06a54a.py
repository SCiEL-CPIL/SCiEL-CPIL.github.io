import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def _make_dense(exps, coeffs):
        """Convert sparse (exps, coeffs) to dense list."""
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        return dense

    def _from_dense(dense):
        """Convert dense list to sparse (exps, coeffs)."""
        exps = []
        coeffs = []
        for i, c in enumerate(dense):
            if c != 0:
                exps.append(i)
                coeffs.append(c)
        return exps, coeffs

    def _crossover(exps1, coeffs1, exps2, coeffs2):
        """
        Crossover: take a random subset of exponents from parent1 and the rest from parent2.
        This creates a hybrid that may have better structure.
        """
        # Sort both parents by exponent
        idx1 = np.argsort(exps1)
        e1 = [exps1[i] for i in idx1]
        c1 = [coeffs1[i] for i in idx1]
        idx2 = np.argsort(exps2)
        e2 = [exps2[i] for i in idx2]
        c2 = [coeffs2[i] for i in idx2]

        # Choose a random split point
        split = random.randint(0, 12)

        # Take first 'split' from parent1, rest from parent2
        new_exps = e1[:split] + e2[split:]
        new_coeffs = c1[:split] + c2[split:]

        # Check for duplicates
        if len(set(new_exps)) != 12:
            # Try another split
            split = random.randint(0, 12)
            new_exps = e1[:split] + e2[split:]
            new_coeffs = c1[:split] + c2[split:]
            if len(set(new_exps)) != 12:
                return None

        # Sort
        idx = np.argsort(new_exps)
        new_exps = [new_exps[i] for i in idx]
        new_coeffs = [new_coeffs[i] for i in idx]

        return new_exps, new_coeffs

    def _mutate(exps, coeffs):
        """
        Mutation: apply 1-3 random perturbations.
        """
        new_exps = exps[:]
        new_coeffs = coeffs[:]
        n_mut = random.randint(1, 3)

        for _ in range(n_mut):
            move_type = random.random()
            if move_type < 0.3:
                # Flip sign of one coefficient
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift one exponent by small amount
                i = random.randint(0, 11)
                delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.7:
                # Jump: random reposition of one exponent
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            else:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                old_c = new_coeffs[i]
                new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                if new_c != old_c and abs(new_c) <= 64:
                    new_coeffs[i] = new_c
                else:
                    continue

        # Validate
        if len(set(new_exps)) != 12:
            return None
        if max(new_exps) > 40:
            return None
        if any(abs(c) > 64 for c in new_coeffs):
            return None
        if any(c == 0 for c in new_coeffs):
            return None

        # Sort
        idx = np.argsort(new_exps)
        new_exps = [new_exps[i] for i in idx]
        new_coeffs = [new_coeffs[i] for i in idx]

        return new_exps, new_coeffs

    def genetic_algorithm(pop_size=40, max_generations=50):
        """
        Genetic algorithm with tournament selection, crossover, and mutation.
        """
        nonlocal best_score, best_candidate

        # Initialize population
        population = []
        scores = []

        # Add best known
        best_exps, best_coeffs = _from_dense(best_candidate)
        population.append((best_exps[:], best_coeffs[:]))
        scores.append(best_score)

        # Add diverse initial candidates
        for _ in range(pop_size - 1):
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            population.append((exps, coeffs))
            scores.append(compute_score_fast(exps, coeffs))

        # Sort by score
        pop_indices = sorted(range(len(population)), key=lambda i: scores[i])
        population = [population[i] for i in pop_indices]
        scores = [scores[i] for i in pop_indices]

        for gen in range(max_generations):
            if time.time() > deadline - 5:
                break

            new_population = []
            new_scores = []

            # Elitism: keep top 5
            for i in range(min(5, len(population))):
                new_population.append(population[i])
                new_scores.append(scores[i])

            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break

                # Tournament selection
                def tournament():
                    indices = random.sample(range(len(population)), min(3, len(population)))
                    best_idx = min(indices, key=lambda i: scores[i])
                    return population[best_idx], scores[best_idx]

                parent1, _ = tournament()
                parent2, _ = tournament()

                # Crossover
                child = _crossover(parent1[0], parent1[1], parent2[0], parent2[1])
                if child is None:
                    # Fallback: just copy parent1
                    child = (parent1[0][:], parent1[1][:])

                # Mutation
                mutated = _mutate(child[0], child[1])
                if mutated is not None:
                    child = mutated
                else:
                    child = (child[0][:], child[1][:])

                score = compute_score_fast(child[0], child[1])
                new_population.append(child)
                new_scores.append(score)

            # Sort new population
            new_pop_indices = sorted(range(len(new_population)), key=lambda i: new_scores[i])
            population = [new_population[i] for i in new_pop_indices]
            scores = [new_scores[i] for i in new_pop_indices]

            # Update best
            if scores[0] < best_score:
                _update_best(population[0][0], population[0][1], scores[0])
                if best_score <= 12:
                    break

        return best_score

    def simulated_annealing(exps, coeffs, max_iters=5000):
        """
        Simulated annealing with diverse moves.
        """
        nonlocal best_score, best_candidate

        # Ensure sorted
        idx = np.argsort(exps)
        exps = [exps[i] for i in idx]
        coeffs = [coeffs[i] for i in idx]

        current_score = compute_score_fast(exps, coeffs)
        _update_best(exps, coeffs, current_score)

        if current_score <= 12:
            return

        # Annealing parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break

            # Choose move type
            move_type = random.random()
            new_exps = exps[:]
            new_coeffs = coeffs[:]

            if move_type < 0.3:
                # Flip sign of one coefficient
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift one exponent by small amount
                i = random.randint(0, 11)
                delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.7:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                nidx = np.argsort(new_exps)
                new_exps = [new_exps[i] for i in nidx]
                new_coeffs = [new_coeffs[i] for i in nidx]
            elif move_type < 0.85:
                # Jump: random reposition of one exponent
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                    nidx = np.argsort(new_exps)
                    new_exps = [new_exps[i] for i in nidx]
                    new_coeffs = [new_coeffs[i] for i in nidx]
                else:
                    continue
            else:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                old_c = new_coeffs[i]
                new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                if new_c != old_c and abs(new_c) <= 64:
                    new_coeffs[i] = new_c
                else:
                    continue

            # Validate
            if len(set(new_exps)) != 12:
                continue
            if max(new_exps) > 40:
                continue
            if any(abs(c) > 64 for c in new_coeffs):
                continue
            if any(c == 0 for c in new_coeffs):
                continue

            new_score = compute_score_fast(new_exps, new_coeffs)

            # Accept or reject
            if new_score < current_score:
                exps = new_exps
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
                if current_score <= 12:
                    break
            else:
                # Simulated annealing acceptance
                delta_score = new_score - current_score
                if T > T_min:
                    accept_prob = np.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        exps = new_exps
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

        return exps, coeffs, current_score

    # --- Main search loop ---

    random.seed(42)

    # Initial candidates
    initial_candidates = []

    # 1. Consecutive exponents with various sign patterns
    for start in range(0, 5):
        exps = [start + i for i in range(12)]
        if max(exps) <= 40:
            # Alternating signs
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            initial_candidates.append((exps[:], coeffs[:]))
            # All positive
            initial_candidates.append((exps[:], [1]*12))
            # Random signs
            for _ in range(5):
                c = [random.choice([-1, 1]) for _ in range(12)]
                initial_candidates.append((exps[:], c))

    # 2. Gapped exponents (e.g., two blocks)
    # Block 1: 0..3, Block 2: 10..13
    exps = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
    if max(exps) <= 40:
        for _ in range(5):
            c = [random.choice([-1, 1]) for _ in range(12)]
            initial_candidates.append((exps[:], c))

    # 3. Random sparse
    for _ in range(20):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        initial_candidates.append((exps, coeffs))

    # Evaluate and run SA on best initial candidates
    scored_candidates = []
    for exps, coeffs in initial_candidates:
        score = compute_score_fast(exps, coeffs)
        if score < float('inf'):
            scored_candidates.append((score, exps, coeffs))
    
    scored_candidates.sort(key=lambda x: x[0])

    # Run SA on the top 5 initial candidates
    for score, exps, coeffs in scored_candidates[:5]:
        if time.time() > deadline - 10:
            break
        simulated_annealing(exps, coeffs, max_iters=2000)
        if best_score <= 12:
            break

    # If still not good, run GA
    if best_score > 12 and time.time() < deadline - 10:
        genetic_algorithm(pop_size=40, max_generations=30)
        if best_score <= 12:
            return best_candidate

    # If still not good, run SA on random restarts
    if best_score > 12 and time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            simulated_annealing(exps, coeffs, max_iters=1000)
            if best_score <= 12:
                break

    return best_candidate

# EVOLVE-BLOCK-END
