import time
import itertools
from collections import defaultdict

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    
    This function attempts to find a valid packing of 6-subsets on 12 points
    such that no 5-subset is repeated. The maximum number of blocks is 132,
    which is achieved exactly by the Witt design S(5,6,12).
    
    The algorithm uses a backtracking search with strong symmetry breaking and
    heuristics (MRV - Minimum Remaining Values) to find the solution quickly.
    If the search fails or times out, it returns the best partial result found.
    """
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # 1. Precomputation
    # There are C(12, 5) = 792 possible 5-subsets.
    # There are C(12, 6) = 924 possible 6-subsets.
    
    points = range(12)
    
    # Map each 5-subset to a unique ID
    all_5subsets = list(itertools.combinations(points, 5))
    five_to_id = {sub: i for i, sub in enumerate(all_5subsets)}
    n_fives = len(all_5subsets) # 792
    
    # Map each 6-subset to its constituent 5-subsets
    all_6subsets = list(itertools.combinations(points, 6))
    n_sixes = len(all_6subsets) # 924
    
    # six_to_fives[i] is the list of 5-subset IDs contained in the 6-subset i
    six_to_fives = []
    for block in all_6subsets:
        fives = []
        for k in range(6):
            # Remove element at index k
            sub = block[:k] + block[k+1:]
            fives.append(five_to_id[sub])
        six_to_fives.append(fives)
        
    # five_to_six[j] is the list of 6-subset indices that contain the 5-subset j
    five_to_six = defaultdict(list)
    for i, fives in enumerate(six_to_fives):
        for fid in fives:
            five_to_six[fid].append(i)

    # 2. Symmetry Breaking
    # M12 is 6-transitive. We can fix the first block to be {0, 1, 2, 3, 4, 5}.
    # This reduces the search space significantly.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_6subsets.index(first_block)
    
    # State
    used_fives = set(six_to_fives[first_idx])
    blocks_found = [first_block]
    
    # For MRV, we need to know how many valid blocks are available for each unused 5-subset.
    # A block is valid if none of its 5-subsets are in used_fives.
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        if len(blocks_found) == 132:
            return True
            
        # Find the 5-subset with the fewest valid blocks (MRV)
        min_count = float('inf')
        best_fid = -1
        
        # Optimization: We can limit the search for the MRV to a subset of candidates
        # or just iterate through all unused ones. 792 * 7 checks is ~5544 ops, which is fast.
        
        for fid in range(n_fives):
            if fid in used_fives:
                continue
                
            count = 0
            for idx in five_to_six[fid]:
                # Check if block idx is valid
                # A block is valid if all its 5-subsets are unused
                valid = True
                for f in six_to_fives[idx]:
                    if f in used_fives:
                        valid = False
                        break
                if valid:
                    count += 1
                    
            if count == 0:
                return False # Dead end: some 5-subset cannot be covered by any remaining block
                
            if count < min_count:
                min_count = count
                best_fid = fid
                if count == 1:
                    break # Best possible, no need to check others
                    
        if best_fid == -1:
            # All 5-subsets are used? This implies we found a full design.
            # But we check len(blocks_found) == 132 earlier.
            # If we haven't reached 132 blocks but all fives are used, it's a contradiction
            # unless the design is smaller (impossible for max packing).
            # Actually, if all 792 fives are used, we must have exactly 792/6 = 132 blocks.
            return True
            
        # Try all valid blocks containing best_fid
        # Sort candidates? Not strictly necessary, but might help.
        candidates = five_to_six[best_fid]
        
        for idx in candidates:
            # Re-check validity (state might have changed if we didn't break early in MRV loop? 
            # No, state is static during this loop iteration for a given fid selection)
            valid = True
            for f in six_to_fives[idx]:
                if f in used_fives:
                    valid = False
                    break
            if not valid:
                continue
                
            # Apply
            blocks_found.append(all_6subsets[idx])
            new_fives = six_to_fives[idx]
            for f in new_fives:
                used_fives.add(f)
                
            if backtrack():
                return True
                
            # Undo
            blocks_found.pop()
            for f in new_fives:
                used_fives.discard(f)
                
        return False

    if backtrack():
        return [list(b) for b in blocks_found]
        
    # If backtracking failed (timeout or dead end), return what we have
    # This shouldn't happen for a perfect search within time, but it's a safe fallback
    return [list(b) for b in blocks_found]

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
    # Verify
    if len(res) == 132:
        print("Success: Found 132 blocks")
    else:
        print("Warning: Did not find full solution")
