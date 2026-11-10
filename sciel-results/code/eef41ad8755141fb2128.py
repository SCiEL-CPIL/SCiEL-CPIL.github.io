import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions
    # Store as tuples of indices (0-based)
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Convert to numpy array for fast vectorized scoring
    # Shape: (num_aps, 4)
    aps_arr = np.array(aps, dtype=np.int32)
    
    def score(colors):
        """
        Calculate the number of monochromatic 4-APs.
        colors: numpy array of shape (N,) with values 0 or 1.
        """
        # Get colors at all AP indices
        # c shape: (num_aps, 4)
        c = colors[aps_arr]
        # Check if all 4 are equal
        # c[:, 0] == c[:, 1] and c[:, 0] == c[:, 2] and c[:, 0] == c[:, 3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 0] == c[:, 2]) & (c[:, 0] == c[:, 3])
        return int(np.sum(mono))

    # Genetic Algorithm Parameters
    pop_size = 60
    mutation_rate = 0.02  # Probability of flipping a bit
    elite_size = 4
    
    # Initialize Population
    # Mix of random and a few structured starts (e.g., alternating, random blocks)
    population = []
    scores = []
    
    # 1. Random individuals
    for _ in range(pop_size - 4):
        ind = np.random.randint(0, 2, size=N, dtype=np.int8)
        population.append(ind)
        scores.append(score(ind))
        
    # 2. Alternating (0,1,0,1...) - often good for avoiding short APs
    alt = np.array([i % 2 for i in range(N)], dtype=np.int8)
    population.append(alt)
    scores.append(score(alt))
    
    # 3. Random blocks of size 4
    block_size = 4
    blocks = np.random.randint(0, 2, size=N // block_size, dtype=np.int8)
    blk = np.repeat(blocks, block_size)
    if len(blk) < N:
        blk = np.concatenate([blk, np.random.randint(0, 2, size=N - len(blk), dtype=np.int8)])
    population.append(blk)
    scores.append(score(blk))
    
    # 4. Random blocks of size 8
    block_size = 8
    blocks = np.random.randint(0, 2, size=N // block_size, dtype=np.int8)
    blk = np.repeat(blocks, block_size)
    if len(blk) < N:
        blk = np.concatenate([blk, np.random.randint(0, 2, size=N - len(blk), dtype=np.int8)])
    population.append(blk)
    scores.append(score(blk))
    
    # Sort initial population
    # We keep a list of (score, individual)
    pop_data = list(zip(scores, population))
    pop_data.sort(key=lambda x: x[0])
    
    best_score = pop_data[0][0]
    best_ind = pop_data[0][1].copy()
    
    # GA Loop
    while time.time() < deadline:
        new_pop_data = []
        
        # 1. Elitism: Keep the best individuals
        for i in range(elite_size):
            s, ind = pop_data[i]
            new_pop_data.append((s, ind.copy()))
        
        # 2. Generate Offspring
        while len(new_pop_data) < pop_size:
            # Selection: Tournament selection (size 3)
            # Pick 3 random individuals from current population, choose the best
            i1 = random.randint(0, len(pop_data) - 1)
            i2 = random.randint(0, len(pop_data) - 1)
            i3 = random.randint(0, len(pop_data) - 1)
            
            candidates = [pop_data[i1], pop_data[i2], pop_data[i3]]
            candidates.sort(key=lambda x: x[0])
            parent1 = candidates[0][1]
            
            i4 = random.randint(0, len(pop_data) - 1)
            i5 = random.randint(0, len(pop_data) - 1)
            i6 = random.randint(0, len(pop_data) - 1)
            
            candidates2 = [pop_data[i4], pop_data[i5], pop_data[i6]]
            candidates2.sort(key=lambda x: x[0])
            parent2 = candidates2[0][1]
            
            # Crossover: Uniform crossover
            # For each bit, pick from parent1 or parent2 with 50% probability
            mask = np.random.randint(0, 2, size=N, dtype=np.int8)
            child = np.where(mask == 1, parent1, parent2)
            
            # Mutation: Flip bits with probability mutation_rate
            # Use a random boolean mask
            mut_mask = np.random.random(N) < mutation_rate
            child[mut_mask] ^= 1  # Flip bits where mask is True
            
            # Evaluate Child
            c_score = score(child)
            new_pop_data.append((c_score, child))
            
            # Check time inside the inner loop to avoid long stalls
            if time.time() > deadline - 1.0:
                break
        
        # Update population
        pop_data = new_pop_data
        
        # Sort new population
        pop_data.sort(key=lambda x: x[0])
        
        # Update best
        if pop_data[0][0] < best_score:
            best_score = pop_data[0][0]
            best_ind = pop_data[0][1].copy()
            
            if best_score == 0:
                break
        
        # Optional: Local search refinement on the best individual every few generations
        # This helps escape shallow local minima that GA might get stuck in
        # But we must be careful with time.
        # Let's do a quick greedy flip on the best individual if we have time.
        if time.time() < deadline - 2.0:
            current_best = best_ind.copy()
            current_score = best_score
            improved = False
            # Simple greedy: try flipping each bit, keep if it improves
            # This is O(N * score_cost). Score cost is vectorized so it's fast.
            # But doing this every generation might be too slow.
            # Let's do it every 10th "generation" (iteration of while loop)
            # We don't have an explicit generation counter, so let's just do it if random chance says so?
            # Or just rely on GA. GA with mutation is usually enough.
            # Let's add a small local search step for the top individual only.
            # Actually, the prompt says "rethinking the approach". GA is the new approach.
            # Adding local search makes it Memetic GA. Let's stick to pure GA first to see if it works.
            # If pure GA is slow, we can optimize.
            
            # However, pure GA can be slow to converge on N=256.
            # Let's implement a very fast local search for the best individual 
            # but only if we have plenty of time left.
            remaining = deadline - time.time()
            if remaining > 5.0:
                # Try to improve best_ind with a few random flips
                # This is effectively a "1+1" EA step on the best
                # But we already have mutation in the GA.
                # Let's just let the GA run.
                pass

    return best_ind.tolist()

# Test execution time and validity
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result length: {len(res)}")
    # Verify output format
    assert len(res) == 256
    assert all(c in [0, 1] for c in res)
    print("Valid")
