import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all t-subsets and map them to indices
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Greedy construction
    cover_count = [0] * n_subsets
    blocks = []
    uncovered = set(range(n_subsets))
    
    # Precompute for each block, the list of subset indices it covers
    # This is expensive: C(19,7) * C(7,4) = 50388 * 35 ~ 1.7M operations. Acceptable.
    # But we can compute on the fly or cache. Let's cache block -> subset_indices
    block_subsets_cache = {}
    def get_block_subsets(block):
        if block not in block_subsets_cache:
            block_subsets_cache[block] = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        return block_subsets_cache[block]
        
    # For greedy, we need to find the block covering most uncovered subsets.
    # Naive: iterate all blocks, count uncovered.
    # Optimization: iterate all blocks, but stop if time is up.
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_count = -1
        # Iterate all possible blocks
        for block in itertools.combinations(range(v), k):
            b_subs = get_block_subsets(block)
            # Count how many of these are in uncovered
            # This is a set intersection
            cnt = len(uncovered.intersection(b_subs))
            if cnt > best_count:
                best_count = cnt
                best_block = block
                if best_count == len(uncovered): # Perfect block
                    break
        
        if best_block is None or best_count == 0:
            # Should not happen if uncovered is not empty and v, k, t are valid
            break
            
        blocks.append(best_block)
        b_subs = get_block_subsets(best_block)
        for idx in b_subs:
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered.discard(idx)
                
    # Local Search
    # We maintain cover_count and the list of blocks.
    # We also need to know which subsets are covered by which blocks?
    # No, cover_count is enough for redundancy check.
    # But for replacement, we need to know if a block contains specific subsets.
    
    def is_redundant(block):
        b_subs = get_block_subsets(block)
        for idx in b_subs:
            if cover_count[idx] <= 1:
                return False
        return True

    def remove_block(block):
        nonlocal blocks, cover_count
        b_subs = get_block_subsets(block)
        for idx in b_subs:
            cover_count[idx] -= 1
        blocks.remove(block)

    def add_block(block):
        nonlocal blocks, cover_count
        b_subs = get_block_subsets(block)
        for idx in b_subs:
            cover_count[idx] += 1
        blocks.append(block)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Reduction Phase
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
                improved = True
                # Do not increment i, as list shifted
            else:
                i += 1
                
        # 2. Optimization Phase
        # Try to replace blocks
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
                
            B = blocks[i]
            b_subs = get_block_subsets(B)
            
            # Find critical subsets (covered only by B)
            critical = [idx for idx in b_subs if cover_count[idx] == 1]
            
            if not critical:
                # Should have been removed
                continue
                
            # Union of elements in critical subsets
            U_B = set()
            for idx in critical:
                U_B.update(all_subsets[idx])
                
            if len(U_B) > k:
                continue
                
            # Find best replacement block B' containing U_B
            # Candidates are blocks containing U_B
            # We want to maximize the number of subsets in B' that are already covered (cover_count > 1)
            # to potentially make other blocks redundant.
            # Or simply maximize the number of subsets B' covers in total?
            # Let's maximize the number of subsets in B' that have cover_count > 1.
            
            best_B_prime = None
            best_score = -1
            
            # Generate candidates containing U_B
            # U_B is a set of points.
            # We need to choose k - len(U_B) points from the remaining v - len(U_B) points.
            remaining_points = [p for p in range(v) if p not in U_B]
            num_to_choose = k - len(U_B)
            
            for candidate_extra in itertools.combinations(remaining_points, num_to_choose):
                B_prime = tuple(sorted(list(U_B) + list(candidate_extra)))
                
                # Skip if B_prime is already in blocks (and not B)
                # Actually, if B_prime is in blocks, replacing B with B_prime is just a swap or no-op.
                # We only care if B_prime is NOT B.
                if B_prime == B:
                    continue
                
                # Calculate score: number of subsets in B_prime with cover_count > 1
                # Note: cover_count currently includes B.
                # If we replace B with B_prime:
                # For s in B_prime:
                #   new_count = cover_count[s] - 1 (if s in B) + 1
                #   if s in B and s in B_prime: new_count = cover_count[s]
                #   if s not in B and s in B_prime: new_count = cover_count[s] + 1
                #   if s in B and s not in B_prime: new_count = cover_count[s] - 1 (must be >= 1, i.e., s not critical)
                
                # We know B_prime contains all critical subsets of B.
                # So for s in critical: s in B and s in B_prime. new_count = cover_count[s] = 1. OK.
                
                b_prime_subs = get_block_subsets(B_prime)
                score = 0
                for idx in b_prime_subs:
                    # If idx is in B, cover_count[idx] includes B.
                    # If idx is not in B, cover_count[idx] does not include B.
                    # We want to know how many subsets B_prime covers that are "well-covered" by others.
                    # A subset s is "well-covered" if cover_count[s] - (1 if s in B else 0) > 1?
                    # No, we want to maximize redundancy.
                    # Let's just count how many subsets in B_prime have cover_count > 1 (current count).
                    # This is a heuristic.
                    if cover_count[idx] > 1:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_B_prime = B_prime
                    
            if best_B_prime is not None:
                # Simulate replacement
                # Check if replacement allows any block to become redundant
                # We can just do it and check.
                
                # Remove B
                b_subs_B = get_block_subsets(B)
                for idx in b_subs_B:
                    cover_count[idx] -= 1
                    
                # Add B_prime
                b_subs_Bp = get_block_subsets(best_B_prime)
                for idx in b_subs_Bp:
                    cover_count[idx] += 1
                    
                blocks[i] = best_B_prime
                
                # Check if any block is now redundant
                redundant_found = False
                for j in range(len(blocks)):
                    if is_redundant(blocks[j]):
                        redundant_found = True
                        break
                        
                if redundant_found:
                    improved = True
                    # We will go back to Reduction phase
                    break
                else:
                    # Revert
                    for idx in b_subs_Bp:
                        cover_count[idx] -= 1
                    for idx in b_subs_B:
                        cover_count[idx] += 1
                    blocks[i] = B

    return blocks
