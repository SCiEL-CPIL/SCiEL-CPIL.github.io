import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
        
    # Pre-compute subset_to_blocks: which blocks contain a given t-subset
    subset_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)
            
    def greedy_cover():
        uncovered = [1] * n_t_subs
        cover_count = [0] * n_t_subs
        blocks = []
        
        # uncovered_count_per_block[i] = number of subsets in block i that are currently uncovered
        uncovered_count_per_block = [len(subs) for subs in block_t_subs]
        total_uncovered = n_t_subs
        
        while total_uncovered > 0:
            if time.time() - start_time > time_limit * 0.4:
                break
                
            # Find best block (max uncovered count)
            max_count = -1
            best_b_idx = -1
            for i in range(n_blocks):
                if uncovered_count_per_block[i] > max_count:
                    max_count = uncovered_count_per_block[i]
                    best_b_idx = i
                    if max_count == len(block_t_subs[i]):
                        break
            if best_b_idx == -1 or max_count == 0:
                break
                
            blocks.append(best_b_idx)
            
            # Update counts
            for s in block_t_subs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in subset_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
        return blocks
    
    def remove_redundant(block_indices):
        if not block_indices:
            return block_indices
        
        cover_count = [0] * n_t_subs
        for b_idx in block_indices:
            for s in block_t_subs[b_idx]:
                cover_count[s] += 1
                
        changed = True
        while changed and (time.time() - start_time) < time_limit * 0.8:
            changed = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for s in block_t_subs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_t_subs[b_idx]:
                        cover_count[s] -= 1
                    block_indices.pop(i)
                    changed = True
                    break
        return block_indices
    
    def local_search(block_indices, max_time=30.0):
        end_time = start_time + max_time
        if time.time() > end_time:
            return block_indices
            
        cover_count = [0] * n_t_subs
        for b_idx in block_indices:
            for s in block_t_subs[b_idx]:
                cover_count[s] += 1
                
        improved = True
        while improved and (time.time() - start_time) < time_limit * 0.9:
            improved = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                critical = [s for s in block_t_subs[b_idx] if cover_count[s] == 1]
                if not critical:
                    continue
                # Find replacement
                candidates = set(subset_to_blocks[critical[0]])
                for s in critical[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                best_gain = 0
                best_new = -1
                for new_idx in candidates:
                    if new_idx == b_idx or new_idx in block_indices:
                        continue
                    loss = 0
                    new_set = set(block_t_subs[new_idx])
                    for s in block_t_subs[b_idx]:
                        if cover_count[s] == 1 and s not in new_set:
                            loss += 1
                    gain = -loss
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_idx
                if best_gain > 0:
                    for s in block_t_subs[b_idx]:
                        cover_count[s] -= 1
                    for s in block_t_subs[best_new]:
                        cover_count[s] += 1
                    block_indices[i] = best_new
                    improved = True
                    break
        return block_indices
    
    def crossover(parent1, parent2):
        # Merge two coverings by picking blocks that cover most uncovered subsets
        covered = [0] * n_t_subs
        for b_idx in parent1:
            for s in block_t_subs[b_idx]:
                covered[s] += 1
        for b_idx in parent2:
            for s in block_t_subs[b_idx]:
                covered[s] += 1
                
        # Start with empty, greedily add blocks from union that cover most uncovered
        uncovered = [0 if covered[s] > 0 else 1 for s in range(n_t_subs)]
        total_uncovered = sum(uncovered)
        blocks = []
        candidate_blocks = set(parent1) | set(parent2)
        
        while total_uncovered > 0 and (time.time() - start_time) < time_limit * 0.9:
            best_b_idx = -1
            best_count = 0
            for b_idx in candidate_blocks:
                count = 0
                for s in block_t_subs[b_idx]:
                    if uncovered[s]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_b_idx = b_idx
            if best_b_idx == -1 or best_count == 0:
                break
            blocks.append(best_b_idx)
            for s in block_t_subs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
        return blocks
    
    def repair(blocks):
        # Ensure validity by adding blocks if needed
        covered = [0] * n_t_subs
        for b_idx in blocks:
            for s in block_t_subs[b_idx]:
                covered[s] += 1
        uncovered = [s for s in range(n_t_subs) if covered[s] == 0]
        while uncovered and (time.time() - start_time) < time_limit * 0.9:
            s = uncovered.pop(0)
            best_b_idx = -1
            best_count = 0
            for b_idx in subset_to_blocks[s]:
                if b_idx in blocks:
                    continue
                count = 0
                for s2 in block_t_subs[b_idx]:
                    if covered[s2] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_b_idx = b_idx
            if best_b_idx == -1:
                break
            blocks.append(best_b_idx)
            for s2 in block_t_subs[best_b_idx]:
                covered[s2] += 1
        return blocks
    
    # Main algorithm
    # 1. Generate initial population via greedy with random tie-breaking
    population = []
    pop_size = 5
    for _ in range(pop_size):
        blocks = greedy_cover()
        if blocks:
            blocks = remove_redundant(blocks)
            population.append(blocks)
    
    if not population:
        return []
    
    best_blocks = min(population, key=len)
    
    # 2. Genetic Algorithm
    max_generations = 10
    for gen in range(max_generations):
        if time.time() - start_time > time_limit * 0.9:
            break
            
        # Selection: keep top half
        population.sort(key=len)
        population = population[:pop_size // 2]
        
        # Crossover to fill population
        while len(population) < pop_size:
            if len(population) < 2:
                population.append(population[0][:])
                continue
            p1 = population[random.randrange(len(population))]
            p2 = population[random.randrange(len(population))]
            child = crossover(p1, p2)
            if child:
                child = remove_redundant(child)
                population.append(child)
        
        # Local search on best few
        for i in range(min(3, len(population))):
            if time.time() - start_time > time_limit * 0.9:
                break
            improved = local_search(population[i], max_time=10.0)
            population[i] = improved
            if len(improved) < len(best_blocks):
                best_blocks = improved
    
    # 3. Final repair and redundancy removal
    best_blocks = remove_redundant(best_blocks)
    best_blocks = repair(best_blocks)
    best_blocks = remove_redundant(best_blocks)
    
    final_blocks = [all_blocks[i] for i in best_blocks]
    return final_blocks

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
