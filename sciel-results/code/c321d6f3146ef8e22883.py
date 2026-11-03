import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin to ensure we return before the 120s hard limit
    deadline = start_time + 110.0

    # Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subs = len(t_subsets)
    
    k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(k_blocks)
    
    # Precompute which t-subsets each k-block covers
    # block_covers[i] is a set of indices of t-subsets covered by k_blocks[i]
    block_covers = []
    for block in k_blocks:
        covered = set()
        for t_sub in itertools.combinations(block, t):
            covered.add(t_sub_to_idx[t_sub])
        block_covers.append(covered)

    best_blocks = None
    best_len = float('inf')

    # Helper: Greedy cover
    def greedy_cover(randomize=False, top_k=1):
        uncovered = set(range(num_t_subs))
        selected_blocks = []
        
        while uncovered:
            max_cov = 0
            best_indices = []
            
            # Find the block that covers the most uncovered subsets
            for i in range(num_k_blocks):
                cov = block_covers[i]
                # Optimization: quick check if any overlap
                # len(cov & uncovered)
                c = len(cov & uncovered)
                if c > max_cov:
                    max_cov = c
                    best_indices = [i]
                elif c == max_cov and c > 0:
                    best_indices.append(i)
            
            if max_cov == 0:
                # Should not happen if a cover exists, but break to avoid infinite loop
                break

            if randomize and len(best_indices) > 1:
                # Pick randomly from the top candidates to diversify
                # top_k is used if we want to pick from top K, but here we just pick from all max
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
            
            selected_blocks.append(idx)
            uncovered -= block_covers[idx]
            
            # Time check inside greedy to avoid overshooting
            if time.time() > deadline - 5.0:
                # Return partial result? No, must be valid.
                # If we break here, the cover might be incomplete.
                # So we should only break the *outer* loop.
                # If we are inside greedy, we must finish it to get a valid cover.
                # So we don't break here, but the outer loop will stop starting new ones.
                pass

        return selected_blocks

    # Helper: Remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Count how many times each t-subset is covered
        counts = [0] * num_t_subs
        for b_idx in blocks:
            for s_idx in block_covers[b_idx]:
                counts[s_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                is_redundant = True
                for s_idx in block_covers[b_idx]:
                    if counts[s_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # This block can be removed
                    for s_idx in block_covers[b_idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            
            blocks = new_blocks
        
        return blocks

    # Main search loop
    while time.time() < deadline - 10.0:
        # Alternate between deterministic and randomized greedy
        if random.random() < 0.3:
            # Randomized greedy
            cand = greedy_cover(randomize=True)
        else:
            # Deterministic greedy
            cand = greedy_cover(randomize=False)
        
        # Refine
        cand = remove_redundant(cand)
        
        if len(cand) < best_len:
            best_len = len(cand)
            best_blocks = cand
            
        # If we hit the known best or lower bound, we can stop
        # Lower bound for C(19,8,5) is 264. Best known is 335.
        if best_len <= 335:
            break

    # Fallback if no valid cover was found (should not happen)
    if best_blocks is None:
        cand = greedy_cover(randomize=False)
        best_blocks = remove_redundant(cand)

    return best_blocks
