import itertools
import random
import time
import copy

def construct(q, n, radius):
    # Specialized for K_3(7,1) but robust for small q^n
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task is specific
        # Simple greedy fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_index = {w: i for i, w in enumerate(all_words)}
        N = len(all_words)
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w); nw[pos] = val
                    neighbors.append(word_index[tuple(nw)])
            balls.append(neighbors)
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in solution: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            solution.append(best)
            for nb in balls[best]:
                uncovered.discard(nb)
        return [all_words[i] for i in solution]

    # Specific logic for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 2187
    
    # Precompute balls (neighbors within distance 1)
    # ball[i] is a list of indices
    balls = []
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                neighbors.append(word_index[tuple(nw)])
        balls.append(neighbors)
        
    # Helper: Check if a set of indices covers all words
    # Returns (is_valid, uncovered_set)
    def check_coverage(indices):
        covered = set()
        for idx in indices:
            covered.update(balls[idx])
        if len(covered) == N:
            return True, set()
        return False, set(range(N)) - covered

    # Helper: Repair a solution to make it valid
    # Adds codewords to cover any uncovered points
    def repair_solution(indices):
        indices = list(indices)
        covered = set()
        for idx in indices:
            covered.update(balls[idx])
        
        uncovered = set(range(N)) - covered
        while uncovered:
            best_idx = -1
            best_count = -1
            # Find codeword covering most uncovered
            # Optimization: Only check candidates not already in solution?
            # Actually, adding an existing one is useless.
            for i in range(N):
                if i in indices: continue
                count = 0
                for nb in balls[i]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            indices.append(best_idx)
            for nb in balls[best_idx]:
                uncovered.discard(nb)
        return indices

    # Helper: Local Search - Remove redundant codewords
    def optimize_solution(indices):
        indices = list(indices)
        # Count coverage for each word
        cover_count = [0] * N
        for idx in indices:
            for nb in balls[idx]:
                cover_count[nb] += 1
                
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            for i in range(len(indices)):
                idx = indices[i]
                # Check if all neighbors of idx are covered by others (count > 1)
                redundant = True
                for nb in balls[idx]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    indices.pop(i)
                    for nb in balls[idx]:
                        cover_count[nb] -= 1
                    changed = True
                    break # Restart loop after removal
        return indices

    # Helper: Swap local search
    # Try replacing one codeword with another to potentially allow further removals
    def swap_local_search(indices):
        indices = list(indices)
        current_size = len(indices)
        improved = True
        while improved:
            improved = False
            # Try all pairs (i, j) where i is in solution, j is not
            # This is O(K * N * K) which might be slow.
            # K ~ 180, N = 2187. 180 * 2187 * 180 ~ 70M ops. Too slow per step.
            # We need a smarter heuristic or limit swaps.
            
            # Instead, let's just try random swaps or limited scope.
            # Given time limits, we might skip complex swaps and rely on GA + Removal.
            # Or, we can do a "1-move" check:
            # For each idx in solution, try replacing it with every other word.
            # Check if the new set is valid and if we can remove more.
            
            # Let's implement a simpler "Add-Remove" move:
            # 1. Remove a codeword.
            # 2. If invalid, add the best codeword to fix.
            # 3. Optimize.
            
            for i in range(len(indices)):
                # Try removing indices[i]
                trial = indices[:]
                trial.pop(i)
                
                # Check validity
                valid, unc = check_coverage(trial)
                if valid:
                    # If valid, it was redundant, optimize should have caught it.
                    # But maybe removal allows other removals?
                    new_trial = optimize_solution(trial)
                    if len(new_trial) < current_size:
                        indices = new_trial
                        current_size = len(indices)
                        improved = True
                        break
                
                # If invalid, try to add one codeword to fix
                if not valid:
                    # Find best codeword to cover most of unc
                    best_add = -1
                    best_count = -1
                    for j in range(N):
                        if j in trial: continue
                        count = sum(1 for nb in balls[j] if nb in unc)
                        if count > best_count:
                            best_count = count
                            best_add = j
                    if best_add != -1:
                        trial.append(best_add)
                        # Now optimize the trial
                        new_trial = optimize_solution(trial)
                        if len(new_trial) < current_size:
                            indices = new_trial
                            current_size = len_indices = len(new_trial)
                            improved = True
                            break
                            
        return indices

    # Genetic Algorithm Setup
    start_time = time.time()
    max_time = 115.0
    
    # Initialize population with diverse solutions
    # 1. Greedy solution
    # 2. Random valid solutions (repaired)
    population = []
    
    # Generate initial greedy solution
    def greedy_init():
        uncovered = set(range(N))
        sol = []
        while uncovered:
            best = -1
            best_count = -1
            # Randomize tie breaking
            candidates = []
            max_c = 0
            for i in range(N):
                if i in sol: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > max_c:
                    max_c = count
                    candidates = [i]
                elif count == max_c and count > 0:
                    candidates.append(i)
            if not candidates: break
            chosen = random.choice(candidates)
            sol.append(chosen)
            for nb in balls[chosen]:
                uncovered.discard(nb)
        return optimize_solution(sol)
        
    pop_size = 20
    for _ in range(pop_size):
        # Mix of greedy and random
        if random.random() < 0.5:
            sol = greedy_init()
        else:
            # Random set of ~200 words, then repair
            k = random.randint(150, 250)
            random_sol = random.sample(range(N), k)
            sol = repair_solution(random_sol)
            sol = optimize_solution(sol)
        population.append(sol)
        
    best_solution = min(population, key=len)
    
    while time.time() - start_time < max_time:
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep best
        new_population.append(best_solution)
        
        while len(new_population) < pop_size:
            # Pick two parents
            p1, p2 = random.sample(population, 2)
            
            # Crossover: Uniform
            # Represent as sets for ease? No, lists are fine.
            # We need to ensure the child is roughly the same size or valid.
            # Simple approach: Take random mix of parents, then repair.
            
            # Determine target size? Or just mix?
            # Let's create a candidate by taking elements from p1 and p2.
            # To avoid explosion, we can limit the size.
            
            # A better crossover for set cover:
            # Start with empty set.
            # For each word in p1, if it covers many uncovered words in the context of p2?
            # Too complex.
            
            # Let's do:
            # 1. Take a random subset of p1 and p2.
            # 2. Repair.
            # 3. Optimize.
            
            # Simpler: Randomly pick a size K between len(p1) and len(p2) + noise.
            # Then pick K words from the union of p1 and p2.
            
            union_parents = set(p1) | set(p2)
            target_size = random.randint(min(len(p1), len(p2)), max(len(p1), len(p2)) + 10)
            target_size = min(target_size, N)
            
            if random.random() < 0.8:
                # Mix from union
                if len(union_parents) >= target_size:
                    child = random.sample(list(union_parents), target_size)
                else:
                    child = list(union_parents)
                    # Fill with random
                    while len(child) < target_size:
                        r = random.randint(0, N-1)
                        if r not in child:
                            child.append(r)
            else:
                # Pure mutation of best
                child = best_solution[:]
                # Add/Remove random
                for _ in range(3):
                    if random.random() < 0.5:
                        r = random.randint(0, N-1)
                        if r not in child:
                            child.append(r)
                    else:
                        if child:
                            child.pop(random.randint(0, len(child)-1))
                            
            # Repair and Optimize
            child = repair_solution(child)
            child = optimize_solution(child)
            
            # Occasional deep local search on the best child
            if random.random() < 0.2:
                 # Try swap local search on this child? 
                 # swap_local_search is expensive. 
                 # Let's just do another round of removal.
                 child = optimize_solution(child)
                 
            new_population.append(child)
            
        # Update best
        for sol in new_population:
            if len(sol) < len(best_solution):
                best_solution = sol
                
        # Update population (keep some diversity)
        # Sort by size
        new_population.sort(key=len)
        # Keep top 80%
        cutoff = int(pop_size * 0.8)
        population = new_population[:cutoff]
        # Fill rest with random mutations of best
        while len(population) < pop_size:
            mut = best_solution[:]
            for _ in range(5):
                if random.random() < 0.5:
                    r = random.randint(0, N-1)
                    if r not in mut:
                        mut.append(r)
                else:
                    if mut:
                        mut.pop(random.randint(0, len(mut)-1))
            mut = repair_solution(mut)
            mut = optimize_solution(mut)
            population.append(mut)
            
    return [all_words[i] for i in best_solution]
