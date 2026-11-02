import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_triples[b_idx] = list of triple indices covered by block b_idx
    block_triples = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[b_idx].append(t_idx)
            triple_to_blocks[t_idx].append(b_idx)
            
    def random_init():
        covered = [False] * num_triples
        num_covered = 0
        blocks = []
        while num_covered < num_triples:
            b_idx = random.randint(0, num_blocks - 1)
            new_cov = 0
            for t_idx in block_triples[b_idx]:
                if not covered[t_idx]:
                    new_cov += 1
            if new_cov > 0:
                blocks.append(b_idx)
                for t_idx in block_triples[b_idx]:
                    if not covered[t_idx]:
                        covered[t_idx] = True
                        num_covered += 1
        return blocks

    def local_search(initial_blocks, max_time=110):
        start_time = time.time()
        current_blocks = list(initial_blocks)
        
        # Build cover_count
        cover_count = [0] * num_triples
        for b_idx in current_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
                
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        def update_cover_counts(blocks):
            nonlocal cover_count
            cover_count = [0] * num_triples
            for b_idx in blocks:
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] += 1

        while time.time() - start_time < max_time:
            # Phase 1: Remove redundant blocks
            changed = True
            while changed:
                changed = False
                random.shuffle(current_blocks)
                for i in range(len(current_blocks) - 1, -1, -1):
                    b_idx = current_blocks[i]
                    redundant = True
                    for t_idx in block_triples[b_idx]:
                        if cover_count[t_idx] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for t_idx in block_triples[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        changed = True
                        if len(current_blocks) < best_size:
                            best_size = len(current_blocks)
                            best_blocks = list(current_blocks)
                            
            # Phase 2: Kick
            if len(current_blocks) > 0:
                # Save current state to revert if kick fails
                saved_blocks = list(current_blocks)
                saved_count = list(cover_count)
                
                # Pick a random block to remove
                rem_idx = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[rem_idx]
                
                # Remove block
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] -= 1
                current_blocks.pop(rem_idx)
                
                # Find uncovered triples
                uncovered = [i for i, c in enumerate(cover_count) if c == 0]
                
                # Greedily add blocks to cover uncovered
                while uncovered:
                    seed_idx = random.choice(uncovered)
                    best_b_idx = -1
                    max_covered = -1
                    candidates = []
                    
                    for b_idx in triple_to_blocks[seed_idx]:
                        # Count how many uncovered it covers
                        count = 0
                        for t_idx in block_triples[b_idx]:
                            if cover_count[t_idx] == 0:
                                count += 1
                        if count > max_covered:
                            max_covered = count
                            candidates = [b_idx]
                        elif count == max_covered:
                            candidates.append(b_idx)
                            
                    if not candidates:
                        break
                        
                    chosen = random.choice(candidates)
                    current_blocks.append(chosen)
                    for t_idx in block_triples[chosen]:
                        cover_count[t_idx] += 1
                        
                    # Update uncovered list
                    uncovered = [i for i, c in enumerate(cover_count) if c == 0]
                    
                # After repair, check if better
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                else:
                    # Revert to saved state if not improved
                    current_blocks = saved_blocks
                    cover_count = saved_count

        return best_blocks

    random.seed(42)
    init_blocks = random_init()
    final_blocks = local_search(init_blocks, max_time=110)
    
    return [list(all_blocks[b_idx]) for b_idx in final_blocks]
