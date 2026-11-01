import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all t-subsets and map to index
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    # Map k-subset to its t-subset indices
    k_sub_to_t_indices = {}
    for k_sub in all_k_subs:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_sub_to_idx[t_sub])
        k_sub_to_t_indices[k_sub] = t_indices
        
    # 1. Greedy Initialization
    count = [0] * num_t_subs
    uncovered = set(range(num_t_subs))
    blocks = []
    
    # For speed in greedy, we can pre-compute for each point x, 
    # which t-subsets contain x? 
    # Or just iterate. v=19 is small.
    
    while uncovered:
        if time.time() > deadline:
            break
        # Pick an uncovered t-subset
        # Randomize or pick first? First is deterministic.
        # To allow different runs, we could shuffle, but let's stick to deterministic greedy first.
        u = min(uncovered)
        u_sub = all_t_subs[u]
        
        best_block = None
        best_gain = -1
        
        # Find a block containing u_sub that maximizes coverage of uncovered
        # Iterate all k-subs containing u_sub?
        # u_sub has t points. We need to add k-t points from the remaining v-t points.
        u_points = set(u_sub)
        remaining_points = [p for p in range(v) if p not in u_points]
        
        # Number of candidates: C(v-t, k-t)
        # For v=19, k=7, t=4: C(15, 3) = 455 candidates.
        # This is very fast.
        
        for new_points in itertools.combinations(remaining_points, k - t):
            candidate = tuple(sorted(u_points | set(new_points)))
            
            # Check if candidate is already in blocks? 
            # Greedy usually adds distinct blocks, but let's be safe.
            # Actually, if it's in blocks, its gain is 0 for new coverage.
            
            gain = 0
            for t_idx in k_sub_to_t_indices[candidate]:
                if t_idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate
        
        if best_block is not None:
            blocks.append(best_block)
            for t_idx in k_sub_to_t_indices[best_block]:
                count[t_idx] += 1
                if count[t_idx] == 1:
                    uncovered.discard(t_idx)
    
    # 2. Redundancy Removal
    def remove_redundant(blocks, count):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for t_idx in k_sub_to_t_indices[blocks[i]]:
                    if count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for t_idx in k_sub_to_t_indices[blocks[i]]:
                        count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        return blocks, count

    blocks, count = remove_redundant(blocks, count)
    
    # 3. Local Search
    best_blocks = blocks
    best_count_val = len(blocks)
    
    while time.time() < deadline:
        improved = False
        
        # Try to swap each block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
                
            B_i = blocks[i]
            
            # Find unique subsets of B_i
            unique_subs = []
            for t_idx in k_sub_to_t_indices[B_i]:
                if count[t_idx] == 1:
                    unique_subs.append(t_idx)
            
            if not unique_subs:
                # Should have been removed
                continue
                
            # Required points
            req_points = set()
            for t_idx in unique_subs:
                req_points.update(all_t_subs[t_idx])
            
            if len(req_points) > k:
                continue # Cannot cover all unique subs with one block
            
            # Generate candidates containing req_points
            remaining = [p for p in range(v) if p not in req_points]
            need = k - len(req_points)
            
            best_candidate = None
            best_score = -1
            
            # Score: number of "critical" subsets (count==1 in other blocks) covered
            # Pre-compute set of critical subsets?
            # Critical subsets are those with count==1.
            # We want candidate to cover many of them to potentially make their owners redundant?
            # Actually, if candidate covers a subset s that is unique to B_j (j!=i),
            # then count[s] becomes 2. B_j loses a unique subset.
            # If B_j loses all its unique subsets, it becomes redundant.
            
            # Let's just maximize the number of unique subsets of OTHER blocks covered.
            # To do this fast, we can just check the count of the subsets in the candidate.
            # If count[t_idx] == 1, it is unique to some block.
            # If we add candidate, count becomes 2.
            
            for new_points in itertools.combinations(remaining, need):
                candidate = tuple(sorted(req_points | set(new_points)))
                
                # Skip if candidate is already in blocks
                if candidate in blocks:
                    continue
                
                # Calculate score
                score = 0
                for t_idx in k_sub_to_t_indices[candidate]:
                    if count[t_idx] == 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                # Perform swap
                # Remove B_i
                for t_idx in k_sub_to_t_indices[B_i]:
                    count[t_idx] -= 1
                blocks.pop(i)
                
                # Add best_candidate
                blocks.append(best_candidate)
                for t_idx in k_sub_to_t_indices[best_candidate]:
                    count[t_idx] += 1
                
                improved = True
                # Restart redundancy removal immediately
                blocks, count = remove_redundant(blocks, count)
                
                if len(blocks) < best_count_val:
                    best_count_val = len(blocks)
                    best_blocks = blocks[:]
                
                # Break out of for loop to restart from beginning
                break
        
        if not improved:
            # No single swap improved things
            # Try random restarts? Or just stop.
            # Given the small size, a few random restarts of the greedy might help.
            # But we are in a local search. Let's try a few random swaps that are "neutral" or slightly worse?
            # Or just break to save time.
            break
            
    return best_blocks
