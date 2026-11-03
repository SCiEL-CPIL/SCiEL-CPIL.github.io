import numpy as np
import time

N = 600
TIME_LIMIT = 115

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

def project_simplex(v, s=1.0):
    """Project vector v onto the simplex sum(v)=s, v>=0."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if np.sum(v) <= 0:
        return np.ones(n) * (s / n)
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    r = np.arange(1, n + 1)
    cond = u - (cssv - 1) / r > 0
    rho = np.where(cond)[0][-1] if np.any(cond) else 0
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    if np.sum(w) < 1e-15:
        return np.ones(n) * (s / n)
    return w * (s / np.sum(w))

def genetic_algorithm(time_limit=110):
    pop_size = 100
    elite_size = 5
    start_time = time.time()
    
    # Initialize population with diverse random starts
    population = []
    # 1. Uniform
    population.append(np.ones(N) / N)
    # 2. Triangular
    x = np.linspace(0, 1, N)
    tri = 1 - np.abs(2 * x - 1)
    population.append(tri / np.sum(tri))
    # 3. Random
    for _ in range(pop_size - 2):
        r = np.random.rand(N)
        r = project_simplex(r, 1.0)
        population.append(r)
        
    population = np.array(population)
    scores = np.array([compute_score(f) for f in population])
    
    # Sort by score (lower is better)
    order = np.argsort(scores)
    population = population[order]
    scores = scores[order]
    
    best_f = population[0].copy()
    best_score = scores[0]
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        new_population = []
        
        # Elitism: keep the best ones
        new_population.append(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.choice(pop_size, 3)
            idx2 = np.random.choice(pop_size, 3)
            p1 = population[idx1][np.argmin(scores[idx1])]
            p2 = population[idx2][np.argmin(scores[idx2])]
            
            # Crossover: Blend
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = (1 - alpha) * p1 + alpha * p2
            
            # Mutation: Random perturbation
            noise = np.random.randn(N) * 0.05
            child1 += noise
            child2 += noise
            
            # Project to simplex
            child1 = project_simplex(child1, 1.0)
            child2 = project_simplex(child2, 1.0)
            
            new_population.append(child1)
            new_population.append(child2)
            
        new_population = np.array(new_population[:pop_size])
        population = new_population
        scores = np.array([compute_score(f) for f in population])
        
        # Sort
        order = np.argsort(scores)
        population = population[order]
        scores = scores[order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_f = population[0].copy()
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Run GA
    f_best, score_best = genetic_algorithm(time_limit=TIME_LIMIT - 5)
    
    # Optional: Local polish with a few gradient steps if time permits
    # But the directive says "do not reuse current designs". 
    # So I will stick to GA only or a very distinct local search if needed.
    # However, GA alone might be slow. Let's see.
    
    # If we have time, we can do a few random restarts of GA?
    # Or just return the GA result.
    
    # To be safe and improve score, let's add a simple local search that is NOT gradient descent.
    # E.g., coordinate descent or hill climbing with random swaps.
    # But GA is the main new design.
    
    # Let's try to improve the GA result with a few iterations of "random coordinate moves".
    # This is distinct from gradient descent.
    if time.time() - start_time < TIME_LIMIT - 2:
        f = f_best.copy()
        current_score = score_best
        for _ in range(500):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            # Pick a random index i and j
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j: continue
            # Move small amount of mass from i to j
            delta = 0.01 * f[i]
            if delta > f[i]: delta = f[i]
            f_new = f.copy()
            f_new[i] -= delta
            f_new[j] += delta
            if f_new[i] < 0: f_new[i] = 0
            # Re-normalize to sum=1
            s = np.sum(f_new)
            if s > 0:
                f_new /= s
            new_score = compute_score(f_new)
            if new_score < current_score:
                f = f_new
                current_score = new_score
                
        if current_score < score_best:
            f_best = f
            score_best = current_score

    return [float(x) for x in f_best]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
