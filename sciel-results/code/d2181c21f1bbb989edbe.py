import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Strategy: Genetic Algorithm
    # Population of vectors h.
    # Objective: Minimize J(h) = |2 N max(conv(h,h)) / sum(h)^2|
    # Note: J(h) is scale invariant.
    
    # Initialize population with diverse "blocky" and "oscillatory" patterns
    # to encourage negative lobes in autocorrelation.
    pop_size = 60
    population = []
    
    # 1. Uniform (baseline)
    population.append(np.ones(N))
    
    # 2. Square wave (period 4: +1, +1, -1, -1)
    sq = np.tile([1, 1, -1, -1], N // 4)
    population.append(sq)
    
    # 3. Square wave (period 2: +1, -1)
    sq2 = np.tile([1, -1], N // 2)
    population.append(sq2)
    
    # 4. Sine wave (low freq)
    t = np.linspace(0, 2 * np.pi, N)
    population.append(1 + 0.5 * np.sin(t))
    
    # 5. Antisymmetric Sine (shifted)
    # h[i] = sin(2*pi*k*i/N). Mean is 0. 
    # Wait, if mean is 0, sum(h) is 0 -> invalid.
    # We need non-zero mean.
    # h[i] = A + B*sin(...)
    # To get negative max in conv, we want the correlation to dip low.
    # A high contrast signal with non-zero mean is good.
    for k in range(1, 10):
        h = 1.0 + 0.8 * np.sin(2 * np.pi * k * t / N)
        population.append(h)
        
    # 6. Random blocky
    rng = np.random.RandomState(42)
    for _ in range(pop_size - len(population)):
        # Random blocks of +1 and -1, with a DC offset to ensure non-zero sum
        blocks = rng.choice([1, -1], size=N//10)
        h = np.repeat(blocks, 10)
        # Add DC offset to make sum non-zero and potentially positive
        # If sum is negative, flip sign (J is even in h? No, J(h) = J(-h) because max(conv(-h,-h))=max(conv(h,h)) and sum(-h)^2=sum(h)^2)
        # Actually, if sum is close to 0, it's bad.
        # Ensure sum is significantly non-zero.
        if abs(np.sum(h)) < 1.0:
            h = h + 1.0 # Shift up
        population.append(h)
        
    # Evaluate initial population
    scores = [compute_value(h) for h in population]
    
    # Keep track of best
    best_h = population[np.argmin(scores)]
    best_val = min(scores)
    
    # GA Loop
    mutation_rate = 0.1
    crossover_rate = 0.7
    elite_count = 5
    offspring_count = pop_size - elite_count
    
    while time.time() < deadline:
        # Sort by score
        indices = np.argsort(scores)
        population = [population[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        # Update best
        if scores[0] < best_val:
            best_val = scores[0]
            best_h = population[0].copy()
            
        # Create next generation
        next_pop = []
        
        # Elitism
        for i in range(elite_count):
            next_pop.append(population[i].copy())
            
        # Generate offspring
        while len(next_pop) < pop_size:
            # Tournament selection
            def select():
                a, b = rng.randint(0, pop_size), rng.randint(0, pop_size)
                if scores[a] < scores[b]:
                    return population[a]
                else:
                    return population[b]
            
            p1 = select()
            p2 = select()
            
            if rng.random() < crossover_rate:
                # Uniform crossover
                mask = rng.random(N) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
                
            # Mutation: Block mutation
            # Pick a random start and length, flip the sign or add noise
            if rng.random() < mutation_rate:
                start = rng.randint(0, N - 10)
                length = rng.randint(5, 50)
                # Flip sign of a block to create high contrast
                child[start:start+length] *= -1
            elif rng.random() < mutation_rate:
                # Add Gaussian noise
                noise = rng.randn(N) * 0.5
                child += noise
                
            # Normalize/Adjust to keep values reasonable (avoid overflow/underflow)
            # Since J is scale invariant, we can normalize to mean 1 or std 1
            # But we must ensure sum != 0.
            s = np.sum(child)
            if abs(s) < 1e-6:
                # If sum is near zero, add a constant
                child += 1.0
            # Optional: Normalize to unit variance to keep numbers stable
            # var = np.var(child)
            # if var > 1e-6:
            #     child = (child - np.mean(child)) / np.sqrt(var) + 1.0
            
            next_pop.append(child)
            
        population = next_pop
        scores = [compute_value(h) for h in population]

    return best_h

# Ensure construct is the only thing called
if __name__ == "__main__":
    h = construct()
    print("Constructed h")
