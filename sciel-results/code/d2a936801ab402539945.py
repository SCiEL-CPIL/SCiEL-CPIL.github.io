import itertools
import time
import numpy as np
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each block, compute which t-subsets it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = [t_index[combo] for combo in itertools.combinations(block, t)]
        block_t_subsets.append(covered)

    # Build reverse mapping
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)
    block_t_sets = [set(lst) for lst in block_t_subsets]
    block_t_arr = [np.array(lst, dtype=np.int32) for lst in block_t_subsets]

    # Symmetry: precompute block as frozenset of points for fast intersection
    block_points = [set(block) for block in k_blocks]

    def run_greedy(weight_mode='inverse', seed=42):
        rng = np.random.default_rng(seed)
        if weight_mode == 'inverse':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts
        elif weight_mode == 'random':
            t_weights = 0.3 / np.maximum(t_cover_count, 1).astype(np.float64) + 0.7 * rng.random(num_t)
        elif weight_mode == 'exp':
            # Exponential weighting: rare t-subsets get much higher weight
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = np.exp(-safe_counts)
        else:
            t_weights = 1.0 / np.maximum(t_cover_count, 1).astype(np.float64)

        gain = np.zeros(num_k, dtype=np.float64)
        for i, covered in enumerate(block_t_subsets):
            if covered:
                gain[i] = np.sum(t_weights[covered])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and time_left() > 30:
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1e10
            best_idx = int(np.argmax(masked_gain))
            best_gain = gain[best_idx]
            if best_gain <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
        return selected

    def local_search(selected, max_time_frac=0.15):
        if not selected:
            return selected
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        selected_set = set(selected)

        # Phase 1: Repeated single-drop
        for _ in range(50):
            if time_left() < 15:
                break
            changed = False
            for si in range(len(selected) - 1, -1, -1):
                if time_left() < 15:
                    break
                block_idx = selected[si]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(block_idx)
                    changed = True
            if not changed:
                break

        # Phase 2: 1-for-1 swaps with improved scoring
        max_rounds = 100 
        for round_num in range(max_rounds):
            if time_left() < 12:
                break
            improved = False
            order = list(range(len(selected)))
            random.shuffle(order)
            for si in order:
                if time_left() < 12:
                    break
                if improved:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                critical = [idx for idx in drop_set if cover_count[idx] == 1]
                if not critical:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    improved = True
                    continue
                
                # Find candidates that cover all critical t-subsets
                candidate_sets = [set(t_to_blocks[idx]) for idx in critical]
                if not candidate_sets:
                    continue
                candidates = candidate_sets[0]
                for cs in candidate_sets[1:]:
                    candidates.intersection_update(cs)
                
                # Filter out already selected blocks
                candidates -= selected_set
                
                if not candidates:
                    continue
                    
                best_candidate = -1
                best_score = float('inf')
                
                for b in candidates:
                    b_set = block_t_sets[b]
                    lost = 0
                    for idx in drop_set:
                        if idx not in b_set and cover_count[idx] == 1:
                            lost += 1
                            if lost > best_score:
                                break
                    if lost > best_score:
                        continue
                            
                    if lost < best_score:
                        best_score = lost
                        best_candidate = b
                        if best_score == 0:
                            break
                
                if best_candidate != -1:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    improved = True
        return selected

    def greedy_repair(selected):
        """Repair coverage by adding blocks to cover any uncovered t-subsets."""
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        
        selected_set = set(selected)
        result = list(selected)
        
        # Find uncovered t-subsets
        uncovered = [i for i in range(num_t) if cover_count[i] == 0]
        
        if not uncovered:
            return result
        
        # Greedily add blocks to cover uncovered t-subsets
        while uncovered and time_left() > 5:
            # Find block that covers most uncovered t-subsets
            best_block = -1
            best_count = 0
            for b in range(num_k):
                if b in selected_set:
                    continue
                count = 0
                for idx in block_t_subsets[b]:
                    if cover_count[idx] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
            
            if best_block == -1 or best_count == 0:
                break
            
            # Add the block
            result.append(best_block)
            selected_set.add(best_block)
            for idx in block_t_subsets[best_block]:
                if cover_count[idx] == 0:
                    cover_count[idx] = 1
                    uncovered.remove(idx)
                else:
                    cover_count[idx] += 1
        
        return result

    def crossover(parent1, parent2):
        """Combine two valid covers by randomly selecting blocks from each parent."""
        if not parent1 or not parent2:
            return list(parent1) if parent1 else list(parent2)
        
        # Shuffle both parents
        p1 = list(parent1)
        p2 = list(parent2)
        random.shuffle(p1)
        random.shuffle(p2)
        
        # Randomly select blocks from each parent
        child = []
        used = set()
        
        for b in p1:
            if b not in used and random.random() < 0.5:
                child.append(b)
                used.add(b)
        
        for b in p2:
            if b not in used and random.random() < 0.5:
                child.append(b)
                used.add(b)
        
        # Repair coverage
        child = greedy_repair(child)
        
        return child

    def mutation(selected):
        """Remove a random block and re-add the minimum-cost set of blocks to restore validity."""
        if not selected or time_left() < 5:
            return list(selected)
        
        result = list(selected)
        
        # Remove a random block
        if random.random() < 0.7:  # 70% chance to remove a block
            idx = random.randint(0, len(result) - 1)
            removed_block = result.pop(idx)
        
        # Repair coverage
        result = greedy_repair(result)
        
        return result

    def genetic_algorithm(initial_solutions, max_time_frac=0.3):
        """Run a genetic algorithm with a population of covering sets."""
        if not initial_solutions:
            return []
        
        # Initialize population
        population = []
        for sol in initial_solutions:
            if sol:
                population.append(list(sol))
        
        if not population:
            return []
        
        # Ensure minimum population size
        min_pop = max(4, len(population))
        while len(population) < min_pop:
            # Duplicate and mutate
            idx = random.randint(0, len(population) - 1)
            new_sol = mutation(population[idx])
            if new_sol:
                population.append(new_sol)
        
        best_solution = min(population, key=len)
        best_len = len(best_solution)
        
        max_generations = 50
        for gen in range(max_generations):
            if time_left() < 10:
                break
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep the best solution
            next_gen.append(list(best_solution))
            
            # Crossover and mutation
            while len(next_gen) < min_pop:
                if time_left() < 5:
                    break
                
                # Select two parents
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                # Crossover
                child = crossover(p1, p2)
                
                # Mutation
                if random.random() < 0.3:
                    child = mutation(child)
                
                if child:
                    next_gen.append(child)
            
            # Update population
            population = next_gen[:min_pop]
            
            # Update best
            for sol in population:
                if len(sol) < best_len:
                    best_len = len(sol)
                    best_solution = list(sol)
        
        return best_solution

    # Run multiple greedy restarts
    best_selected = None
    best_len = float('inf')

    configs = [
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('random', 2024),
        ('exp', 314),
    ]

    initial_solutions = []
    for i, (mode, seed) in enumerate(configs):
        if time_left() < 20:
            break
        sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]
        initial_solutions.append(sel)

    # Run genetic algorithm with the initial solutions
    if time_left() > 30 and initial_solutions:
        ga_result = genetic_algorithm(initial_solutions, 0.3)
        if ga_result and len(ga_result) < best_len:
            best_len = len(ga_result)
            best_selected = ga_result

    # Final polish
    if best_selected and time_left() > 15:
        best_selected = local_search(best_selected, 0.25)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
    return result
