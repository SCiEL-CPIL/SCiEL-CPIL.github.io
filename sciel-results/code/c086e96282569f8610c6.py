import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        """Remove redundant blocks from a valid cover."""
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in global_to_local:
                            i_other = global_to_local[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
                
                elif old_c == 1:
                    pass
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def is_valid_cover(block_indices):
        """Check if the given block indices form a valid cover."""
        if not block_indices:
            return N_t == 0
        covered = set()
        for g in block_indices:
            for si in block_tsubs[g]:
                covered.add(si)
        return len(covered) == N_t

    def local_search_swap(block_indices, max_swaps=5):
        """Try to improve by swapping blocks. Returns improved list or original."""
        if not block_indices:
            return block_indices
        
        # Build coverage count
        cov_count = np.zeros(N_t, dtype=np.int32)
        for g in block_indices:
            for si in block_tsubs[g]:
                cov_count[si] += 1
        
        # Find uncovered t-subsets (should be none if valid)
        uncovered_tsubs = []
        for si in range(N_t):
            if cov_count[si] == 0:
                uncovered_tsubs.append(si)
        
        if uncovered_tsubs:
            return block_indices  # Not a valid cover, can't do local search
        
        best = list(block_indices)
        best_len = len(best)
        
        for _ in range(max_swaps):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Try removing each block and see if we can add a better one
            for idx, g in enumerate(best):
                # Remove block g
                temp = best[:idx] + best[idx+1:]
                
                # Recompute coverage without g
                cov_count2 = np.zeros(N_t, dtype=np.int32)
                for g2 in temp:
                    for si in block_tsubs[g2]:
                        cov_count2[si] += 1
                
                # Find t-subsets that are now uncovered (were only covered by g)
                newly_uncovered = []
                for si in block_tsubs[g]:
                    if cov_count2[si] == 0:
                        newly_uncovered.append(si)
                
                if not newly_uncovered:
                    # Block g was redundant, remove it
                    best = temp
                    best_len -= 1
                    cov_count = cov_count2
                    improved = True
                    break
                
                # Try to find a single block that covers all newly_uncovered
                # This is expensive, so limit the search
                if len(newly_uncovered) <= 3:
                    # Find candidate blocks that cover the most of newly_uncovered
                    best_candidate = None
                    best_coverage = 0
                    for cand_g in random.sample(range(N_blocks), min(500, N_blocks)):
                        cov_new = 0
                        for si in newly_uncovered:
                            if cov_count2[si] == 0 and si in block_tsubs[cand_g]:
                                cov_new += 1
                        if cov_new > best_coverage:
                            best_coverage = cov_new
                            best_candidate = cand_g
                        if best_coverage == len(newly_uncovered):
                            break
                    
                    if best_candidate is not None and best_coverage == len(newly_uncovered):
                        # Check if this swap improves (same length or better)
                        # For now, accept if it maintains validity
                        best = temp + [best_candidate]
                        improved = True
                        break
            
            if not improved:
                break
        
        return best

    def crossover(parent1, parent2):
        """Combine two parent solutions."""
        # Union of blocks
        combined = list(set(parent1 + parent2))
        return combined

    def mutate(block_indices):
        """Randomly mutate a solution by replacing some blocks."""
        if not block_indices:
            return block_indices
        
        result = list(block_indices)
        n_mutations = random.randint(1, 3)
        
        for _ in range(n_mutations):
            if not result:
                break
            # Remove a random block
            idx = random.randint(0, len(result) - 1)
            removed_block = result.pop(idx)
            
            # Add a random new block
            new_block = random.randint(0, N_blocks - 1)
            if new_block not in result:
                result.append(new_block)
        
        return result

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Genetic algorithm
    population = [list(sol_idx)]
    
    # Add some diversity
    for _ in range(3):
        if time.time() > DEADLINE - 10:
            break
        seed = random.randint(0, 2**31)
        cand = run_greedy(seed)
        cand = reduce_solution_fast(cand)
        if cand not in population:
            population.append(list(cand))
    
    gen = 0
    while time.time() < DEADLINE - 5.0:
        gen += 1
        new_population = []
        
        # Selection: keep the best half
        population.sort(key=len)
        survivors = population[:max(1, len(population) // 2)]
        
        # Crossover and mutation
        while len(new_population) < len(population) and time.time() < DEADLINE - 5.0:
            if len(survivors) >= 2:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = crossover(p1, p2)
            else:
                child = list(survivors[0])
            
            # Mutate
            if random.random() < 0.7:
                child = mutate(child)
            
            # Reduce
            child = reduce_solution_fast(child)
            
            # Local search
            if time.time() < DEADLINE - 10:
                child = local_search_swap(child, max_swaps=2)
                child = reduce_solution_fast(child)
            
            new_population.append(child)
        
        # Update best
        for sol in new_population:
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = [all_blocks[i] for i in sol]
        
        population = new_population
        
        # Occasionally add fresh greedy solutions for diversity
        if gen % 5 == 0 and time.time() < DEADLINE - 10:
            seed = random.randint(0, 2**31)
            cand = run_greedy(seed)
            cand = reduce_solution_fast(cand)
            population.append(cand)
            population = population[:10]  # Limit population size
    
    return best_sol
