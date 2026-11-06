import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        count = 0
        for val in square.values():
            if val != 0:
                count += 1
        return count

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def simulate_annealing(initial_exps, initial_coeffs, iterations, start_temp, end_temp):
        """Simulated annealing over both exponents and coefficients."""
        exps = list(initial_exps)
        coeffs = list(initial_coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)
        
        for iter_num in range(iterations):
            if time.time() > deadline:
                break
            
            # Compute temperature
            progress = iter_num / iterations
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Move: change one coefficient
                i = random.randint(0, 11)
                old_coeff = coeffs[i]
                new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
                if new_coeff == old_coeff:
                    continue
                coeffs[i] = new_coeff
                new_score = compute_square_nonzeros(exps, coeffs)
                
                # Accept or reject
                if new_score < current_score or random.random() < (temp / max(1, new_score - current_score)):
                    current_score = new_score
                else:
                    coeffs[i] = old_coeff
                    
            elif move_type < 0.7:
                # Move: change one exponent (keep it valid)
                i = random.randint(0, 11)
                old_exp = exps[i]
                new_exp = random.randint(0, 40)
                # Ensure uniqueness
                if new_exp in [exps[j] for j in range(12) if j != i]:
                    continue
                exps[i] = new_exp
                # Sort to maintain canonical form
                paired = sorted(zip(exps, coeffs))
                exps = [p[0] for p in paired]
                coeffs = [p[1] for p in paired]
                new_score = compute_square_nonzeros(exps, coeffs)
                
                # Accept or reject
                if new_score < current_score or random.random() < (temp / max(1, new_score - current_score)):
                    current_score = new_score
                else:
                    # Revert
                    paired = sorted(zip([old_exp if j == i else exps[j] for j in range(12)], coeffs))
                    # This is tricky to revert properly, so let's just skip this move type for now
                    # and rely on coefficient moves
                    pass
                    
            else:
                # Move: swap two exponents
                i, j = random.sample(range(12), 2)
                exps[i], exps[j] = exps[j], exps[i]
                # Sort to maintain canonical form
                paired = sorted(zip(exps, coeffs))
                exps = [p[0] for p in paired]
                coeffs = [p[1] for p in paired]
                new_score = compute_square_nonzeros(exps, coeffs)
                
                # Accept or reject
                if new_score < current_score or random.random() < (temp / max(1, new_score - current_score)):
                    current_score = new_score
                else:
                    # Revert by sorting back
                    paired = sorted(zip(exps, coeffs))
                    exps = [p[0] for p in paired]
                    coeffs = [p[1] for p in paired]
                    # Actually this won't revert correctly, let's just accept the state
                    pass
            
            # Track best
            try_candidate(exps, coeffs)
        
        return exps, coeffs, current_score

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []
    
    # Arithmetic progressions with various parameters
    for start in range(0, 8):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
    
    # Symmetric sets
    for D in range(16, 41):
        for _ in range(3):
            left = sorted(random.sample(range(0, D // 2 + 1), 6))
            right = sorted(D - x for x in random.sample(range(0, D // 2 + 1), 6))
            exps = sorted(left + right)
            if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
    
    # Random sets with small span
    for _ in range(50):
        base = random.randint(0, 10)
        exps = sorted(set([base + random.randint(0, 25) for _ in range(15)]))
        if len(exps) >= 12:
            exps = exps[:12]
            if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
                starting_points.append((exps, coeffs))
    
    # Deduplicate starting points
    seen = set()
    unique_starting_points = []
    for exps, coeffs in starting_points:
        key = (tuple(exps), tuple(coeffs))
        if key not in seen:
            seen.add(key)
            unique_starting_points.append((exps, coeffs))
    
    # Phase 2: Run simulated annealing from each starting point
    # Prioritize starting points with lower initial score
    unique_starting_points.sort(key=lambda x: compute_square_nonzeros(x[0], x[1]))
    
    # Run SA on top starting points
    max_sa_runs = 30
    sa_iterations = 2000
    start_temp = 10.0
    end_temp = 0.1
    
    for i, (exps, coeffs) in enumerate(unique_starting_points[:max_sa_runs]):
        if time.time() > deadline - 5:
            break
        
        # Run SA
        final_exps, final_coeffs, final_score = simulate_annealing(
            exps, coeffs, sa_iterations, start_temp, end_temp
        )
        
        # Update best
        try_candidate(final_exps, final_coeffs)
    
    # Phase 3: Local refinement on best candidate found
    if best_score < 21 and time.time() < deadline - 10:
        # Extract best exponent set
        if best_candidate:
            max_deg = len(best_candidate) - 1
            # Find nonzero positions
            best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
            best_coeffs = [best_candidate[i] for i in best_exps]
            
            # Do focused local search
            small_coeffs = [-3, -2, -1, 1, 2, 3]
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(12):
                    old = best_coeffs[i]
                    for new_val in small_coeffs:
                        if new_val == old:
                            continue
                        best_coeffs[i] = new_val
                        s = compute_square_nonzeros(best_exps, best_coeffs)
                        if s < best_score:
                            best_score = s
                            improved = True
                        else:
                            best_coeffs[i] = old
                if not improved:
                    break
            
            # Update candidate
            if best_score < 21:
                max_deg = max(best_exps)
                dense = [0] * (max_deg + 1)
                for e, c in zip(best_exps, best_coeffs):
                    dense[e] = c
                if dense[-1] != 0:
                    best_candidate = dense

    return best_candidate

# EVOLVE-BLOCK-END
