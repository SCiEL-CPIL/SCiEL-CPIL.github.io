import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit:
                break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays AND as frozensets for fast intersection
    neighbors = []
    neighbor_sets = []
    for i, w in enumerate(all_words):
        neigh_list = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_list.append(word_to_idx[new_w])
        neighbors.append(np.array(neigh_list, dtype=int))
        neighbor_sets.append(frozenset(neigh_list))
    
    # Build a neighbor-of-neighbor matrix for faster greedy: 
    # For each word i, we want to know which codewords cover i.
    # Actually, let's vectorize the greedy differently.
    
    # Precompute: for each position, for each value, which words have that value at that position
    # A word w is within distance 1 of word x if they differ in at most 1 position.
    # Let's precompute the "ball" of each word as a boolean vector.
    # That's 6561 * 6561 booleans = ~42MB, feasible.
    
    # Actually, let's use a smarter approach: 
    # For greedy, we need to find the word that covers the most uncovered points.
    # Precompute coverage matrix: cover_matrix[i] = boolean array of size N
    # where cover_matrix[i][j] = True if word j is in the ball of word i.
    # This is 6561 * 6561 = 43M booleans ≈ 43MB. OK.
    
    # Build coverage matrix using the neighbor arrays
    cover_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        cover_matrix[i, neighbors[i]] = True
    
    # Start with a greedy construction to get an initial code
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 118.0
    
    while np.any(uncovered):
        # Vectorized: count uncovered points covered by each candidate
        counts = cover_matrix @ uncovered  # dot product gives count of uncovered in each ball
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
    
    def get_counts(code):
        if not code:
            return np.zeros(N, dtype=int)
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts
    
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Genetic Algorithm with local search
    population_size = 6
    population = [(C.copy(), get_counts(C))]
    for _ in range(population_size - 1):
        # Create diverse population members via perturbation
        pert_C = C.copy()
        pert_counts = get_counts(pert_C)
        if len(pert_C) >= 5:
            to_remove = random.sample(pert_C, random.randint(3, 8))
            for c in to_remove:
                pert_C.remove(c)
                pert_counts[neighbors[c]] -= 1
            uncovered = pert_counts < 1
            while np.any(uncovered):
                best_idx = -1
                best_count = -1
                for i in range(N):
                    if i in pert_C:
                        continue
                    cnt = np.sum(uncovered[neighbors[i]])
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_count == 0:
                    break
                pert_C.append(best_idx)
                pert_counts[neighbors[best_idx]] += 1
                uncovered = pert_counts < 1
            pert_C, pert_counts = reduce_code(pert_C, pert_counts)
        population.append((pert_C, pert_counts))
    
    # Ensure best is in population
    best_C, best_counts = reduce_code(C.copy(), get_counts(C.copy()))
    population[0] = (best_C.copy(), best_counts.copy())
    
    def local_search_single(code, counts, time_budget=3.0):
        """Run local search on a single code for a time budget."""
        code = list(code)
        code_set = set(code)
        counts = counts.copy()
        local_start = time.time()
        
        while time.time() - local_start < time_budget:
            # Reduce
            new_code = []
            for c in code:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    code_set.discard(c)
                else:
                    new_code.append(c)
            code = new_code
            if not code:
                break
            
            improved = False
            for c in code[:]:
                if time.time() - local_start >= time_budget:
                    break
                if c not in code:
                    continue
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    continue
                if len(unique_points) > 3:
                    continue
                
                # Find candidate replacement: must cover all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbor_sets[u])
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                candidates_set.difference_update(code_set)
                candidates_set.discard(c)
                if not candidates_set:
                    continue
                
                # Pick the candidate that covers the most already-covered points
                best_c_prime = None
                max_overlap = -1
                for c_prime in candidates_set:
                    overlap = np.sum(counts[neighbors[c_prime]] > 0)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_c_prime = c_prime
                
                if best_c_prime is None:
                    continue
                
                # Try the swap
                old_code = code.copy()
                old_code_set = code_set.copy()
                old_counts = counts.copy()
                code.remove(c)
                code_set.discard(c)
                code.append(best_c_prime)
                code_set.add(best_c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[best_c_prime]] += 1
                
                if np.any(counts < 1):
                    code = old_code
                    code_set = old_code_set
                    counts = old_counts
                    continue
                
                # Reduce after swap
                new_code = []
                for cc in code:
                    if np.min(counts[neighbors[cc]]) >= 2:
                        counts[neighbors[cc]] -= 1
                        code_set.discard(cc)
                    else:
                        new_code.append(cc)
                code = new_code
                improved = True
            
            if not improved:
                break
        
        return code, counts
    
    def repair_code(code):
        """Repair a code to ensure full coverage using vectorized greedy."""
        code_set = set(code)
        counts = get_counts(code)
        uncovered = counts < 1
        while np.any(uncovered):
            # Vectorized greedy
            cand_counts = cover_matrix @ uncovered
            # Mask out already-in-code candidates
            for c in code_set:
                cand_counts[c] = 0
            best_idx = int(np.argmax(cand_counts))
            if cand_counts[best_idx] == 0:
                break
            code.append(best_idx)
            code_set.add(best_idx)
            counts[neighbors[best_idx]] += 1
            uncovered = counts < 1
        # Reduce
        new_code = []
        for c in code:
            if np.min(counts[neighbors[c]]) >= 2:
                counts[neighbors[c]] -= 1
                code_set.discard(c)
            else:
                new_code.append(c)
        return new_code, counts
    
    def crossover(parent1, parent2):
        """Crossover: exchange codewords between two parents."""
        code1 = list(parent1)
        code2 = list(parent2)
        
        if not code1 or not code2:
            return (code1, get_counts(code1)), (code2, get_counts(code2))
        
        # Randomly pick some codewords to swap
        n_swap = random.randint(1, min(3, len(code1), len(code2)))
        idx1 = random.sample(range(len(code1)), n_swap)
        idx2 = random.sample(range(len(code2)), n_swap)
        
        for i in range(n_swap):
            code1[idx1[i]], code2[idx2[i]] = code2[idx2[i]], code1[idx1[i]]
        
        code1, counts1 = repair_code(code1)
        code2, counts2 = repair_code(code2)
        return (code1, counts1), (code2, counts2)
    
    def mutate(code, counts):
        """Mutation: remove random codewords and repair."""
        code = list(code)
        counts = counts.copy()
        if len(code) < 5:
            return code, counts
        
        n_remove = random.randint(2, min(6, len(code)))
        to_remove = random.sample(code, n_remove)
        to_remove_set = set(to_remove)
        new_code = [c for c in code if c not in to_remove_set]
        for c in to_remove:
            counts[neighbors[c]] -= 1
        
        new_code, counts = repair_code(new_code)
        return new_code, counts
    
    # Main GA loop
    generation = 0
    no_improvement = 0
    time_limit_search = time_limit - 3  # 3s safety margin
    
    while time.time() - start_time < time_limit_search:
        # Run local search on each population member
        for i in range(len(population)):
            if time.time() - start_time > time_limit_search:
                break
            code, counts = population[i]
            code, counts = local_search_single(code, counts, time_budget=1.5)
            population[i] = (code, counts)
            if len(code) < best_size:
                best_size = len(code)
                best_C = code.copy()
                best_counts = counts.copy()
                no_improvement = 0
        
        if time.time() - start_time > time_limit_search:
            break
        
        # Crossover: create offspring from pairs
        new_population = []
        for i in range(0, len(population) - 1, 2):
            if time.time() - start_time > time_limit_search:
                break
            offspring1, offspring2 = crossover(population[i][0], population[i+1][0])
            new_population.append(offspring1)
            new_population.append(offspring2)
        
        # Mutation on some members
        for i in range(len(new_population)):
            if time.time() - start_time > time_limit_search:
                break
            if random.random() < 0.4:
                code, counts = mutate(new_population[i][0], new_population[i][1])
                new_population[i] = (code, counts)
        
        # Selection: keep best population_size members
        new_population.sort(key=lambda x: len(x[0]))
        population = new_population[:population_size]
        
        generation += 1
        if len(population[0][0]) < best_size:
            best_size = len(population[0][0])
            best_C = population[0][0].copy()
            best_counts = population[0][1].copy()
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > 15:
            break
    
    # Final polish: aggressive local search on the best code
    if time.time() - start_time < time_limit - 1:
        best_C, best_counts = local_search_single(best_C, get_counts(best_C), time_budget=min(2.0, time_limit - 1 - (time.time() - start_time)))
    
    # Final Reduction
    new_code = []
    counts = get_counts(best_C)
    for c in best_C:
        if np.min(counts[neighbors[c]]) >= 2:
            counts[neighbors[c]] -= 1
        else:
            new_code.append(c)
    best_C = new_code
    
    return [list(all_words[i]) for i in best_C]
