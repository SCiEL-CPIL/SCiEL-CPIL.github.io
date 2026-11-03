import numpy as np
import time
import random

N = 600
TIME_LIMIT = 115  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def normalize(f):
    """Normalize f to have a fixed sum for stable comparison, but keep nonnegative."""
    s = np.sum(f)
    if s < 1e-15:
        return np.ones(N) * (1.0 / N)
    return f / s

def generate_initial_population(pop_size):
    """Generate a diverse initial population."""
    population = []
    # Some random uniform
    for _ in range(pop_size // 4):
        f = np.random.rand(N)
        population.append(f)
    
    # Some blocky structures (fewer distinct values)
    for _ in range(pop_size // 4):
        blocks = np.random.randint(2, 10)
        f = np.zeros(N)
        for i in range(blocks):
            start = int(i * N / blocks)
            end = int((i + 1) * N / blocks)
            val = np.random.rand()
            f[start:end] = val
        population.append(f)
        
    # Some smooth sine waves
    for _ in range(pop_size // 4):
        freq = np.random.randint(1, 5)
        phase = np.random.rand() * 2 * np.pi
        x = np.linspace(0, 2 * np.pi * freq, N)
        f = 0.5 + 0.5 * np.sin(x + phase)
        population.append(f)
        
    # Fill rest with random
    while len(population) < pop_size:
        f = np.random.rand(N)
        population.append(f)
        
    return population

def tournament_selection(population, scores, k=3):
    """Select k individuals randomly and return the best one."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    best_score = scores[best_idx]
    for idx in indices[1:]:
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_idx = idx
    return population[best_idx].copy()

def blend_crossover(parent1, parent2, alpha):
    """Blend crossover: child = alpha * p1 + (1-alpha) * p2"""
    return alpha * parent1 + (1 - alpha) * parent2

def mutate(f, rate, strength):
    """Apply Gaussian mutation to a fraction of elements."""
    f_mut = f.copy()
    mask = np.random.rand(N) < rate
    if np.any(mask):
        noise = np.random.randn(np.sum(mask)) * strength
        f_mut[mask] += noise
    # Ensure nonnegativity
    f_mut = np.maximum(f_mut, 0.0)
    return f_mut

def compute_gradient(f):
    """Compute numerical gradient of the score with respect to f."""
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros(N)
    
    # Autoconvolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    max_val = np.max(conv)
    max_idx = np.argmax(conv)
    
    # The score is max(conv) / (s^2)
    # Gradient with respect to f[i]:
    # d(score)/df[i] = (1/s^2) * d(max_conv)/df[i] - 2*max_conv/s^3 * (1/N)
    # d(max_conv)/df[i] = 2 * sum over j where conv[j] = max of f[j] * f[i-j]
    # This is complex, so we use a simpler approach
    
    # Actually, let's use a different approach: since we want to minimize max(conv)/s^2,
    # we can think of it as minimizing the peak of the convolution while keeping sum fixed.
    # A good heuristic: reduce values at positions that contribute to the peak.
    
    # Find which f values contribute most to the peak
    # conv[k] = sum_i f[i] * f[k-i]
    # For the peak at max_idx, the contributing pairs are (i, max_idx-i)
    
    # Simple gradient: for each i, how much does f[i] affect the peak?
    # If i <= max_idx < i+N, then f[i] contributes to conv[max_idx] via f[max_idx-i]
    # But this is getting complex. Let's use a practical approach.
    
    # Practical approach: the peak is likely at the center (autoconvolution of symmetric-ish functions)
    # We want to flatten the peak. So we should reduce values where they create sharp peaks.
    
    # Alternative: use the fact that for a step function on [-1/4, 1/4],
    # the optimal shape might be related to a triangle or cosine taper.
    
    # For now, return a simple heuristic gradient
    # Reduce values that are in the "middle" more aggressively if the peak is in the middle
    center = N // 2
    dist_from_center = np.abs(np.arange(N) - center)
    # Normalize
    dist_from_center = dist_from_center / max(dist_from_center)
    
    # If the function has high values in the middle, reduce them
    middle_weight = 1.0 - dist_from_center
    middle_weight = middle_weight / np.sum(middle_weight)
    
    # Gradient: push towards uniform in the middle, but keep edges
    # Actually, let's just return a simple perturbation direction
    return -middle_weight * np.sum(f * middle_weight)

def local_hill_climb(f, max_iterations=5000, step_size=0.05):
    """Fast local refinement using coordinate descent and random perturbations."""
    current_f = f.copy()
    current_score = compute_score(current_f)
    
    # Phase 1: Coordinate descent - try adjusting one bin at a time
    for iteration in range(min(max_iterations // 2, 2000)):
        # Pick a random index to adjust
        idx = np.random.randint(N)
        
        # Try several step sizes
        best_new_val = current_f[idx]
        best_new_score = current_score
        
        for delta in [-step_size, -step_size/2, step_size/2, step_size]:
            new_f = current_f.copy()
            new_f[idx] = max(0.0, current_f[idx] + delta)
            new_score = compute_score(new_f)
            if new_score < best_new_score:
                best_new_score = new_score
                best_new_val = new_f[idx]
        
        if best_new_val != current_f[idx]:
            current_f[idx] = best_new_val
            current_score = best_new_score
    
    # Phase 2: Random multi-point perturbations with adaptive step size
    step_size = 0.02
    for iteration in range(max_iterations // 2):
        # Pick a few random indices to perturb
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        new_f = current_f.copy()
        for idx in indices:
            new_f[idx] += np.random.randn() * step_size
            new_f[idx] = max(0.0, new_f[idx])
            
        new_score = compute_score(new_f)
        
        if new_score < current_score:
            current_f = new_f
            current_score = new_score
            step_size *= 0.98
        else:
            step_size *= 0.995
    
    return current_f, current_score

def construct():
    start_time = time.time()
    pop_size = 50
    
    # Initialize population
    population = generate_initial_population(pop_size)
    scores = [compute_score(f) for f in population]
    
    best_f = population[np.argmin(scores)].copy()
    best_score = min(scores)
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    generations = 0
    
    # Main GA loop
    while time.time() - start_time < TIME_LIMIT - 20:
        generations += 1
        new_population = []
        new_scores = []
        
        # Elitism: keep the best individual
        best_idx = np.argmin(scores)
        new_population.append(population[best_idx].copy())
        new_scores.append(scores[best_idx])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
                
            # Select parents
            p1 = tournament_selection(population, scores)
            p2 = tournament_selection(population, scores)
            
            # Crossover
            alpha = np.random.rand()
            child = blend_crossover(p1, p2, alpha)
            
            # Mutation
            mutation_rate = 0.05 + 0.05 * (T / 0.1)  # Higher mutation when T is high
            mutation_strength = 0.1 + 0.1 * (T / 0.1)
            child = mutate(child, mutation_rate, mutation_strength)
            
            # Evaluate
            child_score = compute_score(child)
            
            # Simulated Annealing acceptance
            # Compare with the worst in current population (or random)
            worst_idx = np.argmax(scores)
            if child_score < scores[worst_idx]:
                accept = True
            else:
                delta = child_score - scores[worst_idx]
                if delta < 0:
                    accept = True
                else:
                    accept = np.random.rand() < np.exp(-delta / (T + 1e-12))
            
            if accept:
                new_population.append(child)
                new_scores.append(child_score)
            else:
                # If rejected, just add a mutated version of a random parent to maintain diversity
                random_parent = population[np.random.randint(len(population))]
                rejected_child = mutate(random_parent, 0.1, 0.2)
                rejected_score = compute_score(rejected_child)
                new_population.append(rejected_child)
                new_scores.append(rejected_score)
        
        # Update population
        population = new_population
        scores = new_scores
        
        # Track global best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_f = population[current_best_idx].copy()
            
        # Cool down
        T *= cooling_rate
        
    # Final local refinement with multiple starts
    if time.time() - start_time < TIME_LIMIT - 5:
        # First, refine the best solution
        refined_f, refined_score = local_hill_climb(best_f, max_iterations=8000, step_size=0.05)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
        
        # Try a few more refinements from slightly perturbed versions
        remaining_time = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining_time > 10:
            for attempt in range(3):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                # Create a slightly perturbed version
                perturbed = best_f.copy()
                perturbed += np.random.randn(N) * 0.01
                perturbed = np.maximum(perturbed, 0.0)
                
                refined_f, refined_score = local_hill_climb(perturbed, max_iterations=4000, step_size=0.03)
                if refined_score < best_score:
                    best_f = refined_f
                    best_score = refined_score
    
    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
