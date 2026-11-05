import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    # Beam search
    # State: (sequence, set_of_products)
    # We store the sequence to reconstruct and to know the last element.
    # We store the set of products to check for duplicates.
    
    beam = [([], set())]
    best_seq = []
    
    # We will iterate layer by layer (by length)
    # Or we can just keep expanding the beam until it's empty or we run out of time.
    # Since we want max length, we should expand all states of length L to get states of length L+1.
    
    current_length = 0
    
    while True:
        next_beam = []
        
        for seq, prods in beam:
            if not seq:
                last = 1
                # Candidates 2..128
                start_val = 2
            else:
                last = seq[-1]
                start_val = last + 1
            
            # Optimization: If we have little time, we might not want to check all 128 candidates for every state
            # But 128 is small.
            
            for x in range(start_val, 129):
                # Check if x creates duplicate products
                # New products are x * suffixes of seq
                # suffixes: 1, last, last*second_last, ...
                
                # We need to check if any new product is in prods
                # Also, we need to update prods if we add x
                
                # Let's compute the new suffixes
                # Actually, we can just check intersection.
                
                # Efficient check:
                # The new products are x, x*last, x*last*prev, ...
                # We can generate these and check against prods.
                
                # If x is in prods, invalid.
                if x in prods:
                    continue
                
                # Check longer suffixes
                valid = True
                new_prods_to_add = [x]
                prod = x
                # Iterate backwards through seq
                # seq is [a1, a2, ..., ak]
                # We need x*ak, x*ak*a{k-1}, ...
                # We can just multiply prod by elements of seq in reverse
                
                # To avoid modifying prods in place, we collect new ones
                # and check intersection at the end, or check as we go.
                
                # Let's check as we go for early termination
                # We already checked x.
                
                for a in reversed(seq):
                    prod *= a
                    if prod in prods:
                        valid = False
                        break
                    new_prods_to_add.append(prod)
                    
                if not valid:
                    continue
                
                # If valid, create new state
                new_seq = seq + [x]
                new_prods = prods.union(new_prods_to_add)
                
                next_beam.append((new_seq, new_prods))
                
                if time.time() - start_time > timeout:
                    # Return best found so far
                    if len(new_seq) > len(best_seq):
                        best_seq = new_seq
                    return best_seq
        
        if not next_beam:
            break
            
        # Update best
        # Find the longest sequence in next_beam
        # Since we generate all length L+1 from length L, all in next_beam have same length
        if len(next_beam[0][0]) > len(best_seq):
             # Just pick one? Or keep all in beam.
             # The beam will keep the best ones.
             pass
        
        # Select top K from next_beam
        # Sort by length (all same), then by last element (ascending is better for future)
        # Maybe also by number of "free" candidates? That's expensive.
        # Let's just sort by last element.
        next_beam.sort(key=lambda s: (s[0][-1] if s[0] else 0))
        
        # Keep top K
        K = 2000
        beam = next_beam[:K]
        
        # Check time
        if time.time() - start_time > timeout:
             # Return the longest sequence in the current beam
             if beam:
                 # All have same length, pick the first one (smallest last)
                 return beam[0][0]
             return best_seq

    # If we exited the loop, return the best from the last full beam or best_seq
    # Actually, if next_beam is empty, we can't extend anymore.
    # The best sequence is in the last non-empty beam.
    # Let's track the best sequence explicitly.
    
    # Re-logic:
    # We should return the longest sequence found.
    
    # Let's rewrite the loop to be cleaner.
    
    # The previous code structure was a bit mixed.
    
    pass

# Let's refine the implementation properly.

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # State: (seq, prods)
    # We use a list of states.
    beam = [([], set())]
    best_seq = []
    
    # We expand layer by layer.
    # The beam contains states of the current depth.
    
    while True:
        next_beam = []
        
        for seq, prods in beam:
            last = seq[-1] if seq else 1
            start_val = last + 1
            
            for x in range(start_val, 129):
                # Check validity
                # New products: x, x*last, ...
                
                # Fast check: x in prods?
                if x in prods:
                    continue
                
                # Check other suffixes
                valid = True
                new_prods = [x]
                prod = x
                
                # We need to multiply by seq in reverse
                # seq is [a1, ..., ak]
                # Suffixes are [ak], [ak, a{k-1}], ...
                # So we multiply by ak, then a{k-1}, etc.
                
                # Using a list for new_prods is fine for small k.
                
                for a in reversed(seq):
                    prod *= a
                    if prod in prods:
                        valid = False
                        break
                    new_prods.append(prod)
                    
                if not valid:
                    continue
                
                new_seq = seq + [x]
                new_prods_set = prods.union(new_prods)
                
                next_beam.append((new_seq, new_prods_set))
                
                if time.time() - start_time > timeout:
                    # Emergency exit
                    if len(new_seq) > len(best_seq):
                        best_seq = new_seq
                    return best_seq
        
        if not next_beam:
            break
        
        # Update best_seq if we found longer ones
        # All elements in next_beam have length len(beam[0][0]) + 1
        # So if next_beam is not empty, we have longer sequences.
        # We don't necessarily need to update best_seq here if we keep the beam,
        # but if we crash or timeout, we need a valid result.
        
        # Sort next_beam to keep the most promising states
        # Heuristic: Smaller last element allows more future candidates.
        # Also, maybe fewer "conflicts" in the future? Hard to measure.
        # Let's stick to smaller last element.
        
        next_beam.sort(key=lambda s: s[0][-1])
        
        # Keep top K
        K = 3000
        beam = next_beam[:K]
        
        if time.time() - start_time > timeout:
             if beam:
                 return beam[0][0]
             return best_seq

    # If we break out, the last valid beam is the longest.
    # But wait, if next_beam is empty, the previous beam was the longest.
    # So we should return the best from the previous beam.
    # Actually, if next_beam is empty, we can't extend the current beam.
    # So the current beam (before expansion) contains the longest sequences.
    # But we overwrote beam with next_beam in the previous iteration?
    # No, `beam` holds the current depth. `next_beam` holds depth+1.
    # If `next_beam` is empty, we break. The `beam` variable still holds the states of the previous depth.
    # So we should return the best from `beam`.
    
    if beam:
        # Sort by length (all same) then last element
        # Actually, we just need *a* longest sequence.
        # The first one in the sorted beam is good.
        return beam[0][0]
        
    return []

# Wait, there is a subtle bug.
# If we enter the loop, `beam` is depth 0.
# We generate `next_beam` (depth 1).
# If `next_beam` is not empty, we set `beam = next_beam`.
# Next iteration, we generate `next_beam` (depth 2) from `beam` (depth 1).
# If `next_beam` is empty, we break.
# At this point, `beam` contains depth 1 states.
# So returning `beam[0][0]` is correct.

# However, if we timeout inside the loop, we return `best_seq` or `new_seq`.
# We should ensure `best_seq` is updated properly or we just return the current `new_seq` if it's valid.
# The code `if len(new_seq) > len(best_seq): best_seq = new_seq` is safe.
# But `best_seq` starts as `[]`.
# In the first successful expansion, `new_seq` has length 1. `best_seq` becomes length 1.
# So on timeout, we return a valid sequence.

# One more thing: The problem asks for strictly increasing integers.
# My loop `for x in range(start_val, 129)` ensures `x > last`.
# And `start_val` is `last + 1`. So `x >= last + 1`.
# This ensures strictly increasing.

# Is K=3000 enough?
# The branching factor is up to 127.
# Depth 1: ~127 states.
# Depth 2: ~127 * 127 = 16k states? No, many are pruned.
# But even if 10% survive, that's 1600.
# Depth 3: 1600 * 127 * 0.1 = 20k.
# We keep top 3000.
# This seems reasonable.

# Let's add a small optimization:
# If `x` is in `prods`, skip.
# This is already done.

# What if we use a set for `prods`? Yes, we do.
# Union operation `prods.union(new_prods)` creates a new set.
# This is O(N). N is size of prods, which is ~k^2/2.
# For k=10, N=50. Very fast.

# Final check on constraints:
# "Return strictly increasing integers in [2,128]."
# "Products of every nonempty consecutive block must be distinct."
# "Maximize length."

# The algorithm maximizes length by exploring all valid extensions up to a beam width limit.
# If the true optimal length is L, and the beam width is large enough to contain at least one path to length L, it will find it.
# Given the constraints, the space is not astronomically large.

# One potential issue: "singleton products are included".
# This means if $x_i$ appears in the sequence, the product $x_i$ is in the set.
# My code adds `x` to `new_prods` and thus to `new_prods_set`.
# And it checks `if x in prods`.
# So if $x_i = x_j$ for $i \neq j$, it would be caught?
# No, the sequence is strictly increasing, so $x_i \neq x_j$.
# But could $x_i$ equal a product of a block involving $x_j$?
# Yes. E.g., sequence [2, 3]. Products: 2, 3, 6.
# Next candidate 4.
# New products: 4, 4*3=12, 4*3*2=24.
# Check against {2, 3, 6}. No conflict.
# Next candidate 5.
# New products: 5, 5*4=20, 5*4*3=60, 5*4*3*2=120.
# Check against {2, 3, 6, 4, 12, 24}. No conflict.
# Sequence [2, 3, 4, 5].
# Next candidate 6.
# New products: 6, 6*5=30, 6*5*4=120...
# 6 is in {2, 3, 6, ...}. Conflict!
# So 6 is rejected.
# This logic is correct.

# The code looks solid.

import time

def construct():
    start_time = time.time()
    timeout = 55.0  # Leave 5s margin for the evaluator's 60s limit
    
    # State: (sequence, set_of_products)
    beam = [([], set())]
    
    while True:
        next_beam = []
        
        for seq, prods in beam:
            last = seq[-1] if seq else 1
            start_val = last + 1
            
            for x in range(start_val, 129):
                # Check if x itself conflicts
                if x in prods:
                    continue
                
                # Check if x * suffixes conflict
                valid = True
                new_prods = [x]
                prod = x
                
                # Multiply by elements of seq in reverse order
                for a in reversed(seq):
                    prod *= a
                    if prod in prods:
                        valid = False
                        break
                    new_prods.append(prod)
                
                if not valid:
                    continue
                
                new_seq = seq + [x]
                new_prods_set = prods.union(new_prods)
                
                next_beam.append((new_seq, new_prods_set))
                
                if time.time() - start_time > timeout:
                    return new_seq
        
        if not next_beam:
            break
        
        # Sort by last element (ascending) to prioritize states with more room
        next_beam.sort(key=lambda s: s[0][-1])
        
        # Keep top K
        K = 3000
        beam = next_beam[:K]
        
        if time.time() - start_time > timeout:
            if beam:
                return beam[0][0]
            return []

    # If we break out, return the best from the last full beam
    if beam:
        return beam[0][0]
    return []

# Wait, if `next_beam` is empty, we break.
# `beam` still holds the states from the previous iteration (which were valid and longest).
# So `return beam[0][0]` is correct.

# One edge case: What if the optimal sequence is just [2]?
# Beam 0: [([], set())]
# Next beam: [([2], {2}), ([3], {3}), ...]
# Sorted by last element: [2] is first.
# Beam 1: [([2], {2}), ...]
# Next iteration: Expand [2]. Candidates 3..128.
# If 3 is valid, add [2,3].
# If no candidates are valid for any state in beam, next_beam is empty.
# Break.
# Return beam[0][0] which is [2] (or whatever was at the top).
# This is correct.

# The score is the length of the sequence.
# The evaluator measures "raw value (higher is better)".
# The task says "Maximize length".
# So returning the list of integers is the correct interface.
# The "value" is likely `len(result)`.

# Let's double check the "value" definition.
# "The evaluator measures the raw value (higher is better)."
# "Return strictly increasing integers..."
# Usually, for such tasks, the score is the length.
# The prompt says "search score is value/seed".
# Seed program returns a list.
# If the seed program returns a list of length L, and the value is L, then score = L/L_seed.
# If my program returns a list of length L_new, score = L_new / L_seed.
# So I just need to return the list.

# Final check on performance.
# 3000 states.
# Each state expands to ~100 candidates.
# 300,000 checks per layer.
# Each check involves a set lookup (O(1)) and a small loop (O(k)).
# 300,000 * 10 ops = 3M ops.
# Python can do ~10M-100M simple ops/sec.
# So 1 layer takes ~0.1s.
# We can do ~500 layers? No, depth is limited by 128.
# Max depth is 127 (if we could pick all).
# But we prune heavily.
# Realistically, we won't go deeper than 15-20.
# So total time will be well under 1 second.
# The timeout guard is just for safety.

# One minor optimization:
# `prods.union(new_prods)` creates a new set.
# We can do `new_prods_set = set(prods)` and then `new_prods_set.update(new_prods)`.
# This might be slightly faster or slower. `union` is implemented in C, so it's likely fast.
# Let's stick with `union`.

# Is there any risk of integer overflow?
# Python integers have arbitrary precision. No risk.

# Is there any risk of the sequence not being strictly increasing?
# `x` ranges from `last + 1` to `128`.
# So `x > last`.
# `last` is the previous element.
# So strictly increasing is guaranteed.

# Is there any risk of out-of-bounds?
# `range(start_val, 129)` stops at 128.
# So elements are in [2, 128].
# Guaranteed.

# The solution seems robust.
