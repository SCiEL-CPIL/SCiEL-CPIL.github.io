import numpy as np
import time

N = 100

def construct():
    # The task is to minimize max overlap of h with shifted 1-h.
    # Overlap for shift s: sum(h[i] * (1 - h[(i+s)%N]))
    # Let's use a Genetic Algorithm approach on a discrete representation.
    # We will represent h as an array of integers in [0, M] and divide by M.
    # M=1 (binary) might be too coarse, M=10 or M=100 is better.
    # However, the sum constraint sum(h) = N/2 is strict.
    # For binary {0,1}, sum must be N/2, so exactly N/2 ones.
    # Let's try a GA on binary strings with exactly N/2 ones first, 
    # then maybe refine with a local search on the continuous values 
    # using the binary pattern as a guide or just stick to discrete if it's better.
    # Actually, the problem says "step heights", implying piecewise constant, 
    # but the input is just a sequence. 
    # Let's try a GA on discrete values {0, 0.5, 1} or just {0,1}.
    # If we use {0,1}, the sum constraint is easy: exactly N/2 ones.
    
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Fast overlap calculation
    def max_overlap_binary(h):
        # h is a numpy array of 0s and 1s
        # overlap_s = sum(h[i] * (1 - h[i+s]))
        # This is equivalent to:
        # sum(h) - sum(h[i]*h[i+s])
        # Since sum(h) is fixed (N/2), minimizing max overlap is equivalent to 
        # maximizing min(sum(h[i]*h[i+s])) over all shifts s?
        # Wait. 
        # Overlap_s = sum(h[i]) - sum(h[i]h[i+s]) = N/2 - Autocorr_s
        # We want to MINIMIZE max(Overlap_s).
        # This is equivalent to MINIMIZING max(N/2 - Autocorr_s)
        # Which is equivalent to MAXIMIZING min(Autocorr_s).
        # So we want to maximize the minimum autocorrelation.
        
        # Autocorrelation for shift s: sum(h[i] * h[(i+s)%N])
        # We can compute all shifts efficiently using FFT or just loop for N=100.
        # N=100 is small, loop is fine.
        
        min_autocorr = float('inf')
        for s in range(1, N):
            shifted = np.roll(h, s)
            ac = np.dot(h, shifted)
            if ac < min_autocorr:
                min_autocorr = ac
        return N/2 - min_autocorr  # This is the max overlap value
        
    # Actually, let's just compute max overlap directly to be safe with the definition.
    # Overlap_s = sum(h[i] * (1 - h[(i+s)%N]))
    
    def calc_max_overlap(h):
        max_ov = 0.0
        for s in range(1, N):
            shifted = np.roll(h, s)
            ov = np.sum(h * (1 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    # Initial Population: Random binary strings with exactly N/2 ones
    pop_size = 200
    rng = np.random.default_rng(42)
    
    def random_binary():
        # Create an array with N/2 ones and N/2 zeros, then shuffle
        arr = np.zeros(N, dtype=np.float64)
        indices = rng.choice(N, N//2, replace=False)
        arr[indices] = 1.0
        return arr

    population = [random_binary() for _ in range(pop_size)]
    scores = [calc_max_overlap(ind) for ind in population]
    
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # If we want to try continuous refinement later, we can, 
    # but let's see how well the binary GA does first.
    # We will run a simple GA with tournament selection and crossover.
    
    generations = 500
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        # Evaluate new individuals if any (we re-evaluate after crossover/mutation)
        # Actually, we keep a population of (h, score) pairs.
        
        next_pop = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        sorted_indices = np.argsort(scores)
        for i in range(elite_count):
            idx = sorted_indices[i]
            next_pop.append((population[idx], scores[idx]))
            
        # Fill rest with offspring
        while len(next_pop) < pop_size:
            # Tournament selection
            i1 = rng.integers(0, pop_size)
            i2 = rng.integers(0, pop_size)
            if scores[i1] < scores[i2]:
                p1 = population[i1]
            else:
                p1 = population[i2]
                
            i3 = rng.integers(0, pop_size)
            i4 = rng.integers(0, pop_size)
            if scores[i3] < scores[i4]:
                p2 = population[i3]
            else:
                p2 = population[i4]
                
            # Crossover: Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: Flip bits
            # We must maintain exactly N/2 ones.
            # Strategy: Pick a 1 and a 0 and swap them, or just flip two bits.
            # Let's do 2-point mutation: pick two distinct indices, one 1 and one 0?
            # Or just pick two random indices and swap their values? 
            # Swapping values in a binary array preserves the count of ones.
            if rng.random() < 0.5:
                i_mut1 = rng.integers(0, N)
                i_mut2 = rng.integers(0, N)
                if i_mut1 != i_mut2:
                    # Swap values
                    temp = child[i_mut1]
                    child[i_mut1] = child[i_mut2]
                    child[i_mut2] = temp
            
            # Calculate score
            score = calc_max_overlap(child)
            next_pop.append((child, score))
            
            # Update best
            if score < best_score:
                best_score = score
                best_h = child.copy()
                
        # Update population
        population = [p[0] for p in next_pop]
        scores = [p[1] for p in next_pop]
        
        # Optional: Local Search on the best individual found so far
        # If we have time, try to improve best_h by flipping pairs of bits
        # that reduce the max overlap.
        if time.time() < deadline - 20:
            # Simple local search on best_h
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                # Try all possible swaps (i, j) where h[i]=1 and h[j]=0
                ones = np.where(best_h == 1.0)[0]
                zeros = np.where(best_h == 0.0)[0]
                
                # To save time, don't try all O(N^2) swaps every step, 
                # but for N=100, N^2/4 = 2500 swaps is fast.
                for i in ones:
                    for j in zeros:
                        # Try swap
                        old_i = best_h[i]
                        old_j = best_h[j]
                        best_h[i] = old_j
                        best_h[j] = old_i
                        
                        new_score = calc_max_overlap(best_h)
                        
                        if new_score < best_score - 1e-12:
                            best_score = new_score
                            improved = True
                            # Restart loop with new best
                            ones = np.where(best_h == 1.0)[0]
                            zeros = np.where(best_h == 0.0)[0]
                            break
                        else:
                            # Revert
                            best_h[i] = old_i
                            best_h[j] = old_j
                    if improved:
                        break
                        
            # Update population with the improved best
            # Replace worst individual with best_h
            worst_idx = np.argmax(scores)
            population[worst_idx] = best_h.copy()
            scores[worst_idx] = best_score

    return best_h

# Keep the function name and signature
def construct():
    return _construct_impl()

# Wrapper to match the required interface if needed, but usually construct is the main one.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program had `def construct():`
# I defined `construct` above. I should remove the wrapper if it's not needed.
# Let's just stick to one `construct` function.

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    N = 100
    
    # Fast overlap calculation
    def calc_max_overlap(h):
        max_ov = 0.0
        for s in range(1, N):
            shifted = np.roll(h, s)
            ov = np.sum(h * (1 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    # Initial Population: Random binary strings with exactly N/2 ones
    pop_size = 200
    rng = np.random.default_rng(42)
    
    def random_binary():
        arr = np.zeros(N, dtype=np.float64)
        indices = rng.choice(N, N//2, replace=False)
        arr[indices] = 1.0
        return arr

    population = [random_binary() for _ in range(pop_size)]
    scores = [calc_max_overlap(ind) for ind in population]
    
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    best_score = scores[best_idx]
    
    generations = 500
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        next_pop = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        sorted_indices = np.argsort(scores)
        for i in range(elite_count):
            idx = sorted_indices[i]
            next_pop.append((population[idx], scores[idx]))
            
        # Fill rest with offspring
        while len(next_pop) < pop_size:
            # Tournament selection
            i1 = rng.integers(0, pop_size)
            i2 = rng.integers(0, pop_size)
            if scores[i1] < scores[i2]:
                p1 = population[i1]
            else:
                p1 = population[i2]
                
            i3 = rng.integers(0, pop_size)
            i4 = rng.integers(0, pop_size)
            if scores[i3] < scores[i4]:
                p2 = population[i3]
            else:
                p2 = population[i4]
                
            # Crossover: Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: Swap two bits to preserve sum
            if rng.random() < 0.5:
                i_mut1 = rng.integers(0, N)
                i_mut2 = rng.integers(0, N)
                if i_mut1 != i_mut2:
                    temp = child[i_mut1]
                    child[i_mut1] = child[i_mut2]
                    child[i_mut2] = temp
            
            # Calculate score
            score = calc_max_overlap(child)
            next_pop.append((child, score))
            
            # Update best
            if score < best_score:
                best_score = score
                best_h = child.copy()
                
        # Update population
        population = [p[0] for p in next_pop]
        scores = [p[1] for p in next_pop]
        
        # Local Search on the best individual found so far
        if time.time() < deadline - 20:
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                ones = np.where(best_h == 1.0)[0]
                zeros = np.where(best_h == 0.0)[0]
                
                for i in ones:
                    for j in zeros:
                        old_i = best_h[i]
                        old_j = best_h[j]
                        best_h[i] = old_j
                        best_h[j] = old_i
                        
                        new_score = calc_max_overlap(best_h)
                        
                        if new_score < best_score - 1e-12:
                            best_score = new_score
                            improved = True
                            ones = np.where(best_h == 1.0)[0]
                            zeros = np.where(best_h == 0.0)[0]
                            break
                        else:
                            best_h[i] = old_i
                            best_h[j] = old_j
                    if improved:
                        break
                        
            worst_idx = np.argmax(scores)
            population[worst_idx] = best_h.copy()
            scores[worst_idx] = best_score

    return best_h
