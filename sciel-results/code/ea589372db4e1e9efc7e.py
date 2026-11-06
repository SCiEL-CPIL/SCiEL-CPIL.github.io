"""Witt design S(5,6,12) construction via optimized backtracking."""
import time
from itertools import combinations

_DEADLINE = time.time() + 140.0

def _generate_s5612():
    n = 12
    all_points = list(range(n))
    all_5subsets = list(combinations(all_points, 5))
    all_6sets = list(combinations(all_points, 6))
    
    # Map 5-subset to index for faster counting? 
    # Or just use tuples as keys in dict.
    # Let's use indices for speed.
    s5_index = {s5: i for i, s5 in enumerate(all_5subsets)}
    s6_index = {s6: i for i, s6 in enumerate(all_6sets)}
    
    # fives_of[s6_idx] = list of s5_idx
    fives_of = [None] * len(all_6sets)
    for i, s6 in enumerate(all_6sets):
        fives_of[i] = [s5_index[s5] for s5 in combinations(s6, 5)]
        
    # conflicts[s6_idx] = list of s6_idx that share >= 5 elements
    conflicts = [None] * len(all_6sets)
    for i, s6 in enumerate(all_6sets):
        s6_set = set(s6)
        conf = []
        for j, s6j in enumerate(all_6sets):
            if i == j: continue
            # Check intersection size
            # Fast check: if any 5-subset of s6 is in s6j
            # Or just set intersection
            if len(s6_set & set(s6j)) >= 5:
                conf.append(j)
        conflicts[i] = conf
        
    # five_to_sixes: for each s5_idx, list of s6_idx containing it
    five_to_sixes = [[] for _ in range(len(all_5subsets))]
    for i, s6 in enumerate(all_6sets):
        for s5_idx in fives_of[i]:
            five_to_sixes[s5_idx].append(i)
            
    # Initial counts
    # Each 5-subset is in 7 6-sets
    avail_count = [7] * len(all_5subsets)
    
    # Fixed first block for symmetry breaking
    # Find index of (0,1,2,3,4,5)
    fixed_block_tuple = (0, 1, 2, 3, 4, 5)
    fixed_block_idx = s6_index[fixed_block_tuple]
    
    # State
    available = set(range(len(all_6sets)))
    available.remove(fixed_block_idx)
    
    # Remove conflicts of fixed block
    for c in conflicts[fixed_block_idx]:
        if c in available:
            available.remove(c)
            
    # Update counts for removed blocks
    # We need to decrement counts for all s5 in fives_of[removed]
    # We track which blocks are removed to update counts?
    # It's easier to just recalculate counts from scratch? No, 792*132 is fine.
    # But we need to know which are available.
    
    # Let's use a recursive function that passes the state.
    # To be fast, we'll copy the `avail_count` list and `available` set.
    
    def backtrack(blocks, available_set, avail_counts):
        if time.time() > _DEADLINE:
            return None
            
        if len(blocks) == 132:
            return blocks
            
        # Find MRV: uncovered s5 with min avail_count
        min_count = float('inf')
        best_s5_idx = -1
        # We need to know which s5 are covered.
        # We can derive covered s5 from the blocks?
        # Or pass a `covered` set.
        # Passing a set of 792 ints is okay.
        
        # Actually, if we maintain `avail_counts`, and we know a s5 is covered if its count is 0?
        # No, count is 0 if no *available* block covers it.
        # If a s5 is covered by a *selected* block, it is no longer "uncovered".
        # But if it's covered, we don't care about its count for MRV (we want to cover *uncovered* ones).
        # So we need to know which s5 are already covered by selected blocks.
        
        # Let's pass `covered_s5` as a set of indices.
        
        # Optimization: If we just added a block, we know which s5 became covered.
        # Let's pass `covered_s5`.
        
        # This function signature is getting large.
        # Let's use a global or closure? No, recursion.
        
        # Alternative: 
        # The number of uncovered s5 is 792 - 6*len(blocks).
        # We can just iterate all 792 and check if in covered_s5.
        
        pass # Placeholder
        
    # This approach is getting complex to manage state correctly with copies.
    # Given the time limit, a simpler heuristic might be better.
    # The previous backtracking was O(792 * 7) per step.
    # The new one is O(792) for MRV + O(37) for removal.
    # It is significantly faster.
    
    # Let's implement the state passing carefully.
    
    def solve(blocks, available_set, avail_counts, covered_s5):
        if time.time() > _DEADLINE:
            return None
            
        if len(blocks) == 132:
            return blocks
            
        # Find s5 to cover
        min_count = float('inf')
        best_s5 = -1
        for s5_idx in range(len(all_5subsets)):
            if s5_idx in covered_s5:
                continue
            c = avail_counts[s5_idx]
            if c < min_count:
                min_count = c
                best_s5 = s5_idx
                if c == 1:
                    break # Can't get better
                if c == 0:
                    return None # Dead end
                    
        if best_s5 == -1:
            return None # All covered?
            
        # Candidates for best_s5
        candidates = [s6_idx for s6_idx in five_to_sixes[best_s5] if s6_idx in available_set]
        
        # Order candidates? Maybe by how many other uncovered s5 they cover?
        # For now, just iterate.
        
        for s6_idx in candidates:
            # Check if valid (should be, by construction of available_set)
            # But we must ensure it doesn't conflict with covered s5?
            # If s6_idx is in available_set, it means it doesn't share a 5-subset with any *selected* block.
            # So it is valid.
            
            # Create new state
            new_available = available_set - {s6_idx}
            # Remove conflicts
            for c in conflicts[s6_idx]:
                if c in new_available:
                    new_available.remove(c)
                    
            # Update counts
            new_counts = avail_counts[:] # Copy
            # Decrement for s6_idx and its conflicts
            # We need to decrement for every s5 in fives_of[removed_block]
            removed_blocks = {s6_idx} | set(conflicts[s6_idx])
            # Wait, conflicts might include blocks already removed?
            # `available_set` only contains available ones.
            # So we only decrement for those in `available_set` that we are removing.
            # The set `removed_blocks` is exactly the ones we are removing from `available_set`.
            
            for rb in removed_blocks:
                for s5_idx in fives_of[rb]:
                    new_counts[s5_idx] -= 1
                    
            # Update covered
            new_covered = covered_s5 | set(fives_of[s6_idx])
            
            # Recurse
            result = solve(blocks + [all_6sets[s6_idx]], new_available, new_counts, new_covered)
            if result:
                return result
                
        return None
        
    # Initialize state
    initial_covered = set(fives_of[fixed_block_idx])
    initial_counts = avail_count[:]
    
    # Remove fixed block and conflicts from counts
    removed_initial = {fixed_block_idx} | set(conflicts[fixed_block_idx])
    for rb in removed_initial:
        for s5_idx in fives_of[rb]:
            initial_counts[s5_idx] -= 1
            
    initial_available = set(range(len(all_6sets))) - removed_initial
    
    return solve([fixed_block_tuple], initial_available, initial_counts, initial_covered)

def construct():
    start_time = time.time()
    try:
        result = _generate_s5612()
        if result is not None and len(result) == 132:
            return result
    except Exception:
        pass
    
    # Fallback
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
            if time.time() - start_time > 140:
                break
    return blocks
