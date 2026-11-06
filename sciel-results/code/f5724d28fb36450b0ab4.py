import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0  # safety margin before 120s
    
    # Precompute all 4-term APs as index tuples (0-based)
    # AP: (i, i+d, i+2d, i+3d) where i+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    def count_mono(colors):
        """Count monochromatic 4-term APs. colors is 0/1 array of length n."""
        # For each AP, check if all 4 are same color
        # colors[aps] gives shape (num_aps, 4)
        c = colors[aps]
        # monochromatic if all equal: c[:,0]==c[:,1]==c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def delta_flip(i, colors):
        """Compute change in mono count if we flip position i."""
        # Find all APs containing position i
        # An AP (a, a+d, a+2d, a+3d) contains i if i == a, a+d, a+2d, or a+3d
        # For each role, find corresponding APs
        changes = 0
        c = colors[aps]
        # Current mono status
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        # For each AP containing i, determine if flipping i changes mono status
        # Position i can be at index 0,1,2,3 in the AP tuple
        # We need to find which APs have i in any position
        
        # Efficient approach: for each of the 4 positions in an AP, find APs where that position == i
        for pos in range(4):
            mask = aps[:, pos] == i
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            for idx in idxs:
                # Check if this AP is currently mono
                was_mono = mono[idx]
                # After flip, check if it would be mono
                new_c = c[idx].copy()
                new_c[pos] = 1 - new_c[pos]
                now_mono = (new_c[0] == new_c[1]) & (new_c[1] == new_c[2]) & (new_c[2] == new_c[3])
                if now_mono and not was_mono:
                    changes += 1
                elif was_mono and not now_mono:
                    changes -= 1
        return changes
    
    def local_search(colors, max_iter=5000):
        """Hill climbing with random restarts within iterations."""
        best_score = count_mono(colors)
        best_colors = colors.copy()
        cur = colors.copy()
        cur_score = best_score
        no_improve = 0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Pick random position to flip
            i = random.randint(0, n - 1)
            d = delta_flip(i, cur)
            if d < 0:
                cur[i] = 1 - cur[i]
                cur_score += d
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_colors = cur.copy()
            else:
                no_improve += 1
                # Occasional random kick to escape local minima
                if no_improve > 50:
                    for _ in range(3):
                        j = random.randint(0, n - 1)
                        cur[j] = 1 - cur[j]
                    cur_score = count_mono(cur)
                    no_improve = 0
        return best_colors, best_score
    
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def block_pattern(block_size):
        return np.array([(i // block_size) % 2 for i in range(n)], dtype=np.int8)
    
    def random_init():
        return np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int8)
    
    # Generate diverse initial solutions
    inits = []
    inits.append(thue_morse())
    inits.append(block_pattern(2))
    inits.append(block_pattern(3))
    inits.append(block_pattern(4))
    inits.append(block_pattern(5))
    inits.append(block_pattern(7))
    inits.append(block_pattern(8))
    inits.append(block_pattern(16))
    # Mod-based patterns
    for mod in [3, 4, 5, 7, 11, 13, 17, 19, 23]:
        inits.append(np.array([(i % mod) < mod // 2 for i in range(n)], dtype=np.int8))
    # Random inits
    for _ in range(5):
        inits.append(random_init())
    
    # Run local search on each initial solution
    best_overall = None
    best_overall_score = float('inf')
    
    for init in inits:
        if time.time() > deadline:
            break
        res_colors, res_score = local_search(init, max_iter=3000)
        if res_score < best_overall_score:
            best_overall_score = res_score
            best_overall = res_colors
    
    # Genetic algorithm with segment crossover
    if time.time() < deadline - 5:
        pop_size = 12
        # Initialize population with best + variations
        population = [best_overall.copy() for _ in range(pop_size // 2)]
        for _ in range(pop_size - len(population)):
            population.append(random_init())
        
        # Evaluate initial population
        pop_scores = [count_mono(p) for p in population]
        
        gen = 0
        while time.time() < deadline - 2:
            gen += 1
            # Sort by score
            order = np.argsort(pop_scores)
            population = [population[i] for i in order]
            pop_scores = [pop_scores[i] for i in order]
            
            if pop_scores[0] < best_overall_score:
                best_overall_score = pop_scores[0]
                best_overall = population[0].copy()
            
            new_pop = []
            # Elitism: keep top 2
            new_pop.append(population[0].copy())
            new_pop.append(population[1].copy())
            
            while len(new_pop) < pop_size:
                if time.time() > deadline - 1:
                    break
                # Select parents (tournament)
                p1_idx = random.choice(range(min(6, len(population))))
                p2_idx = random.choice(range(min(6, len(population))))
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                # Segment crossover: pick 2-4 segments from one parent, rest from other
                child = p1.copy()
                num_segments = random.randint(2, 4)
                pos = 0
                while pos < n and num_segments > 0:
                    seg_len = random.randint(8, 32)
                    end = min(pos + seg_len, n)
                    child[pos:end] = p2[pos:end]
                    pos = end + random.randint(0, 10)
                    num_segments -= 1
                
                # Mutation: flip a few bits
                for _ in range(random.randint(1, 5)):
                    j = random.randint(0, n - 1)
                    child[j] = 1 - child[j]
                
                new_pop.append(child)
            
            population = new_pop
            pop_scores = [count_mono(p) for p in population]
            
            # Local search on the best individual each generation
            if time.time() < deadline - 5:
                res_colors, res_score = local_search(population[0], max_iter=1000)
                if res_score < pop_scores[0]:
                    population[0] = res_colors
                    pop_scores[0] = res_score
                if res_score < best_overall_score:
                    best_overall_score = res_score
                    best_overall = res_colors
    
    # Final local search on the best
    if time.time() < deadline - 1:
        res_colors, res_score = local_search(best_overall, max_iter=5000)
        if res_score < best_overall_score:
            best_overall_score = res_score
            best_overall = res_colors
    
    return list(best_overall)
