import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    # Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # Precompute coverage: block_cover[i] is list of t-subset indices covered by block i
    # Also block_bitset[i] is an integer bitset
    block_cover = []
    block_bitset = []
    for b in k_blocks:
        covers = [t_subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covers)
        bitset = 0
        for idx in covers:
            bitset |= (1 << idx)
        block_bitset.append(bitset)

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << num_t_subsets) - 1
        solution = []
        
        while uncovered:
            best_gain = 0
            best_indices = []
            
            # Find blocks with max gain
            for i in range(num_k_blocks):
                gain = (uncovered & block_bitset[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                # Should not happen
                break
                
            # Randomly pick from best candidates to diversify
            chosen = random.choice(best_indices)
            solution.append(chosen)
            uncovered &= ~block_bitset[chosen]
            
            # Time check inside loop? 
            # If we are very close to deadline, we might not finish.
            # But greedy is fast. 150 blocks * 50k checks = 7.5M ops.
            # 7.5M bit_count ops is < 1 second in Python? 
            # Let's assume it's fast.
            
        return solution

    def local_search_reduce(solution):
        if not solution:
            return solution
            
        # Initialize counts
        counts = [0] * num_t_subsets
        for b_idx in solution:
            for t_idx in block_cover[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            new_solution = []
            for b_idx in solution:
                # Check if removable
                removable = True
                for t_idx in block_cover[b_idx]:
                    if counts[t_idx] == 1:
                        removable = False
                        break
                if removable:
                    for t_idx in block_cover[b_idx]:
                        counts[t_idx] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            solution = new_solution
            
        return solution

    best_solution = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        sol = greedy_construct(seed)
        sol = local_search_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
        # If we reach a very good bound, we can stop
        if best_size <= 135:
            break
            
    if best_solution is None:
        # Fallback: return all blocks if we failed to find a valid one (shouldn't happen)
        return [list(b) for b in k_blocks]
        
    return [list(k_blocks[i]) for i in best_solution]
