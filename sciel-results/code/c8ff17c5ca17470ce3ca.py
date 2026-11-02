import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subs)}
    num_t = len(t_subs)
    full_mask = (1 << num_t) - 1
    
    k_subs = list(itertools.combinations(range(v), k))
    block_masks = []
    for ks in k_subs:
        mask = 0
        for ts in itertools.combinations(ks, t):
            mask |= (1 << t_to_idx[ts])
        block_masks.append(mask)
        
    num_blocks = len(k_subs)
    
    # 2. Greedy Initialization
    def greedy_init():
        uncovered = full_mask
        blocks = []
        while uncovered:
            best_block = -1
            best_gain = -1
            for i in range(num_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            if best_gain == 0:
                break
            blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
        return blocks
        
    # 3. Local Search
    def local_search(blocks, max_iters=10):
        if not blocks:
            return blocks
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            coverage = 0
            for b in blocks:
                coverage |= block_masks[b]
                
            for i in range(len(blocks)):
                b_i = blocks[i]
                rest_cov = 0
                for j in range(len(blocks)):
                    if j != i:
                        rest_cov |= block_masks[j]
                        
                lost = block_masks[b_i] & ~rest_cov
                if lost == 0:
                    blocks.pop(i)
                    improved = True
                    break
                    
                current_gain = (block_masks[b_i] & ~coverage).bit_count()
                best_swap = -1
                best_swap_gain = current_gain
                
                # Scan for better swap
                for j in range(num_blocks):
                    if j in blocks:
                        continue
                    mask_j = block_masks[j]
                    if (mask_j & lost) != lost:
                        continue
                    new_gain = (mask_j & ~coverage).bit_count()
                    if new_gain > best_swap_gain:
                        best_swap_gain = new_gain
                        best_swap = j
                        
                if best_swap != -1:
                    blocks[i] = best_swap
                    improved = True
                    break
                    
        return blocks

    # 4. GA
    pop_size = 20
    g_sol = greedy_init()
    population = [g_sol]
    
    for _ in range(pop_size - 1):
        sol = list(g_sol)
        if sol and random.random() < 0.5:
            idx = random.randint(0, len(sol)-1)
            new_block = random.randint(0, num_blocks-1)
            if new_block not in sol:
                sol[idx] = new_block
        population.append(sol)
        
    best_sol = g_sol
    best_score = len(best_sol)
    
    def get_fitness(blocks):
        if not blocks:
            return (0, 0)
        cov = 0
        for b in blocks:
            cov |= block_masks[b]
        uncovered = (full_mask & ~cov).bit_count()
        return (len(blocks), uncovered)
    
    while time.time() < deadline:
        scored_pop = []
        for ind in population:
            nb, unc = get_fitness(ind)
            score = -1000 * unc - nb
            scored_pop.append((score, ind))
            
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        nb, unc = get_fitness(scored_pop[0][1])
        if unc == 0 and nb < best_score:
            best_score = nb
            best_sol = scored_pop[0][1]
            
        new_pop = [scored_pop[0][1]]
        
        while len(new_pop) < pop_size:
            p1 = random.choice(scored_pop[:min(10, len(scored_pop))])[1]
            p2 = random.choice(scored_pop[:min(10, len(scored_pop))])[1]
            
            child = list(set(p1) | set(p2))
            
            if child and random.random() < 0.5:
                idx = random.randint(0, len(child)-1)
                new_block = random.randint(0, num_blocks-1)
                child[idx] = new_block
            
            child = local_search(child, max_iters=5)
            new_pop.append(child)
            
        population = new_pop
        
    nb, unc = get_fitness(best_sol)
    if unc > 0:
        best_sol = greedy_init()
        
    return [list(k_subs[i]) for i in best_sol]
