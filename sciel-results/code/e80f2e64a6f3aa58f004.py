"""
Strategy: Evolutionary Algorithm (Genetic Algorithm) with Simulated Annealing
and Fast Vectorized Evaluation.

Rationale:
1. The objective function involves convolution and quadratic forms, which are
   smooth but non-convex. Coordinate descent (current strategy) is slow and
   prone to local optima.
2. Genetic Algorithms maintain a population, allowing for global exploration
   and exploitation of diverse regions of the search space.
3. We use a fast vectorized evaluation function to handle many candidates
   quickly within the time budget.
4. We initialize the population with a mix of parametric shapes (Gaussian,
   triangular, uniform) and random noise to cover different "basins of attraction".
"""

import numpy as np
import time
import random

N = 50

def evaluate_batch(h_batch):
    """
    Evaluate a batch of height vectors.
    h_batch: shape (batch_size, N)
    Returns: array of values for each candidate.
    """
    h_batch = np.clip(h_batch, 0, None)
    batch_size = h_batch.shape[0]
    values = np.zeros(batch_size)
    
    # Pre-compute constants
    # g = convolve(h, h) -> length 2N-1
    # m = 2N-1
    # y = [0, g..., 0] -> length 2N+1
    # w = 1 / (m + 1) = 1 / (2N)
    
    m = 2 * N - 1
    w = 1.0 / (m + 1)
    
    # Vectorized convolution is not directly available in standard NumPy for batches
    # without loops or scipy. However, N=50 is small.
    # We can use a loop for convolution but vectorize the rest.
    # Or use scipy.signal.convolve for each? No, let's stick to a tight loop
    # for convolution since N is small, but vectorize the l2/norm calc.
    
    # Actually, let's just loop over batch items but keep the inner ops fast.
    # Or better: use np.convolve in a loop is O(B * N^2). B can be ~100-500.
    # N^2 = 2500. 500 * 2500 = 1.25M ops. This is very fast in Python if vectorized per item?
    # No, np.convolve is C-optimized. A Python loop of 500 calls to np.convolve is fine.
    
    for i in range(batch_size):
        h = h_batch[i]
        g = np.convolve(h, h)
        
        # y construction
        # y[0]=0, y[1:m+1]=g, y[m+1]=0
        # We need sum over i in 0..m of (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
        
        # Let's compute the terms explicitly to avoid constructing large y arrays if possible
        # y has length m+2.
        # Terms for i=0: y[0]=0, y[1]=g[0] -> 0 + 0 + g[0]^2 = g[0]^2
        # Terms for i=m: y[m]=g[m-1], y[m+1]=0 -> g[m-1]^2 + 0 + 0 = g[m-1]^2
        # Terms for i in 1..m-1: y[i]=g[i-1], y[i+1]=g[i]
        # Sum = g[0]^2 + g[m-1]^2 + sum_{k=0}^{m-2} (g[k]^2 + g[k]g[k+1] + g[k+1]^2)
        
        # Let's just construct y, it's cheap.
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        y_i = y[:m+1]
        y_ip1 = y[1:m+2]
        
        terms = y_i**2 + y_i * y_ip1 + y_ip1**2
        l2 = (w / 3.0) * np.sum(terms)
        
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 * norm_inf == 0:
            values[i] = -1.0
        else:
            values[i] = l2 / (norm_1 * norm_inf)
            
    return values

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # 1. Initial Population Generation
    pop_size = 200
    population = np.zeros((pop_size, N))
    
    # Mix of initializers
    # 1. Uniform
    population[0] = 1.0
    # 2. Gaussian centered
    x = np.linspace(-1, 1, N)
    population[1] = np.exp(-4 * x**2)
    # 3. Triangular
    population[2] = np.abs(1 - 2 * np.linspace(0, 1, N))
    # 4. Sine bump
    population[3] = np.sin(np.pi * np.linspace(0, 1, N))
    # 5. Bimodal
    population[4] = np.exp(-10*(x-0.4)**2) + np.exp(-10*(x+0.4)**2)
    # 6. Random uniform
    for i in range(5, pop_size):
        if i % 3 == 0:
            population[i] = np.random.uniform(0, 1, N)
        elif i % 3 == 1:
            # Sparse spikes
            h = np.zeros(N)
            idx = np.random.choice(N, size=10, replace=False)
            h[idx] = np.random.uniform(0.5, 1.0, 10)
            population[i] = h
        else:
            # Smooth random (low pass filter of noise)
            h = np.random.uniform(0, 1, N)
            kernel = np.ones(5)/5
            population[i] = np.convolve(h, kernel, mode='same')
            
    # Evaluate initial population
    values = evaluate_batch(population)
    best_idx = np.argmax(values)
    best_h = population[best_idx].copy()
    best_val = values[best_idx]
    
    # 2. Evolutionary Loop
    # Parameters
    elite_size = 10
    mutation_rate = 0.3
    crossover_rate = 0.8
    annealing_alpha = 0.995
    
    t = 1.0  # Temperature for simulated annealing acceptance
    
    while time.time() < deadline:
        # Sort population by value
        sorted_indices = np.argsort(values)[::-1]
        sorted_pop = population[sorted_indices]
        sorted_vals = values[sorted_indices]
        
        # Update best
        if sorted_vals[0] > best_val:
            best_val = sorted_vals[0]
            best_h = sorted_pop[0].copy()
            
        # Generate next generation
        new_population = np.zeros((pop_size, N))
        new_values = np.zeros(pop_size)
        
        # Keep elites
        new_population[:elite_size] = sorted_pop[:elite_size]
        new_values[:elite_size] = sorted_vals[:elite_size]
        
        # Fill rest with offspring
        for i in range(elite_size, pop_size):
            # Select parents (tournament selection)
            # Pick 3 random, take best
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            p3_idx = np.random.randint(0, pop_size)
            
            v1, v2, v3 = values[p1_idx], values[p2_idx], values[p3_idx]
            parent1_idx = p1_idx
            if v2 > v1: parent1_idx = p2_idx
            if v3 > values[parent1_idx]: parent1_idx = p3_idx
            
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            p3_idx = np.random.randint(0, pop_size)
            
            v1, v2, v3 = values[p1_idx], values[p2_idx], values[p3_idx]
            parent2_idx = p1_idx
            if v2 > v1: parent2_idx = p2_idx
            if v3 > values[parent2_idx]: parent2_idx = p3_idx
            
            p1 = population[parent1_idx]
            p2 = population[parent2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.random(N) < 0.5
                child = np.where(mask, p1, p2)
            else:
                # Blend crossover
                alpha = random.uniform(0, 1)
                child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation
            if random.random() < mutation_rate:
                # Gaussian noise
                child += np.random.normal(0, 0.2, N)
                
            # Occasionally do a "jump" mutation (reinitialize part of vector)
            if random.random() < 0.1:
                idx = np.random.choice(N, size=5, replace=False)
                child[idx] = np.random.uniform(0, 1, 5)
                
            new_population[i] = np.clip(child, 0, None)
            
        # Evaluate new population (only the new ones)
        new_offspring_vals = evaluate_batch(new_population[elite_size:])
        new_values[elite_size:] = new_offspring_vals
        
        # Simulated Annealing Step:
        # Allow worse solutions to stay in population with probability exp(-delta/T)
        # We compare new population with old population?
        # Standard GA replaces. To add SA flavor, we can accept a child
        # if it's better, or with prob exp((val_child - val_parent)/T)
        # Since we generate new pop from scratch, let's do a simple replacement
        # but keep the best of (old, new) for each slot? No, that's messy.
        
        # Instead, let's just replace, but we can tune mutation rate based on T.
        # As T drops, mutation variance can drop.
        
        population = new_population
        values = new_values
        
        # Cool down
        t *= annealing_alpha
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final Local Refinement on the best candidate found
    # Use a quick coordinate descent to polish the best GA result
    def local_refine(h, time_limit):
        h = np.array(h, dtype=np.float64)
        best_h = h.copy()
        best_val = evaluate_batch(best_h.reshape(1, -1))[0]
        t0 = time.time()
        step = 0.05
        for _ in range(1000):
            if time.time() > time_limit:
                break
            i = random.randint(0, N-1)
            for delta in [step, -step, step*0.5, -step*0.5]:
                new_h = best_h.copy()
                new_h[i] = max(0, new_h[i] + delta)
                val = evaluate_batch(new_h.reshape(1, -1))[0]
                if val > best_val:
                    best_val = val
                    best_h = new_h
                    break
        return best_h, best_val
    
    if time.time() < deadline - 10:
        best_h, best_val = local_refine(best_h, deadline - 5)
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
