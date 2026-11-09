import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Bitset of uncovered subsets (1 = uncovered, 0 = covered)
    uncovered = np.ones(n_t_subs, dtype=np.uint8)
    
    blocks = []
    
    # Precompute for each point x, the list of t-subsets containing x
    # This is useful for the final update, but for gain calculation we need something else.
    
    # For gain calculation:
    # We need to count uncovered subsets of the form (S \ {x}) U {x} where S is in the block?
    # No, the block is being built. Let current block be B.
    # We want to add x. The new t-subsets covered are those T such that T subset B U {x}, |T|=t, and x in T.
    # These are T = (T \ {x}) U {x}, where T \ {x} is a (t-1)-subset of B.
    # So for a fixed B, the gain of x is the number of (t-1)-subsets S of B such that S U {x} is uncovered.
    
    # To do this fast:
    # 1. Generate all (t-1)-subsets of B.
    # 2. For each (t-1)-subset S, we need to check all x not in B.
    #    This is still a loop over x.
    #    
    # Alternative Vectorization:
    # Let's precompute for each (t-1)-subset S, a bitmask of points x such that S U {x} is a valid t-subset.
    # Actually, simpler:
    # For a given B, we want to score each x in [0, v) \ B.
    # Score(x) = sum_{S in C(B, t-1)} 1 if (S U {x}) is uncovered.
    # 
    # We can precompute a matrix M where M[i, x] = 1 if the t-subset at index i contains x.
    # Then for a given B, we want to sum over i such that (subset i) is uncovered and (subset i) \ {x} is a subset of B.
    # This condition is equivalent to: subset i is a union of a (t-1)-subset of B and {x}.
    
    # Let's try a different angle.
    # Precompute for each point x, the indices of all t-subsets containing x.
    # subsets_containing_x[x] = list of indices i such that x in all_t_subs[i]
    
    # But we need to intersect with "subsets whose other t-1 elements are in B".
    
    # Let's stick to the (t-1)-subset enumeration but vectorize the lookup.
    # For a given B:
    # 1. Get all (t-1)-subsets of B. Let this be list of tuples S.
    # 2. For each S, we want to find x such that S U {x} is uncovered.
    #    This is equivalent to finding x such that idx(S U {x}) is in the set of uncovered indices.
    
    # Even better:
    # Precompute a dictionary mapping (t-1)-tuple to a list of (x, index) pairs? No, too big.
    
    # Let's use the property:
    # Gain(x) = count of uncovered subsets T such that x in T and T \ {x} subset B.
    
    # We can precompute for each (t-1)-subset S (of the whole universe), the list of x such that S U {x} is a t-subset.
    # Actually, for a fixed S, the valid x are just those not in S.
    # And the index is subset_to_idx(tuple(sorted(S + (x,)))).
    
    # Let's precompute a sparse structure or just use the direct lookup in the loop if it's fast enough?
    # The loop is over (t-1)-subsets of B. B has size up to k=7. t=4.
    # So we iterate over C(7, 3) = 35 subsets.
    # For each subset S (size 3), we need to check all x not in B (size 12).
    # Total checks: 35 * 12 = 420 lookups in a dictionary/array.
    # This is very fast in Python if we use a list for uncovered and direct indexing.
    
    # So the bottleneck is not the gain calculation itself, but the number of blocks and the overhead.
    # The original code was slow because it iterated over ALL uncovered subsets to calculate gain?
    # "sum(1 for rest in itertools.combinations(block, t - 1) if tuple(sorted(rest + (x,))) in uncovered)"
    # Yes, it iterated over combinations of the current block.
    # My proposed method does exactly that, but using a precomputed index map for O(1) lookup.
    
    # Let's implement this optimized greedy.
    
    # Precompute index for any t-subset
    # We already have subset_to_idx.
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Pick first uncovered subset to start the block
        first_idx = np.argmax(uncovered)
        first_sub = all_t_subs[first_idx]
        block = list(first_sub)
        
        # Mark it as covered? No, wait until the whole block is formed.
        # But we can use it as the seed.
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # Calculate gain for each x not in block
            # Gain(x) = number of (t-1)-subsets S of block such that S U {x} is uncovered.
            
            # Optimization:
            # Instead of checking all x, we can iterate over (t-1)-subsets S of block.
            # For each S, we know which x can complete it to a t-subset (any x not in S).
            # But we only care about x not in block.
            
            # Let's accumulate scores for x.
            scores = np.zeros(v, dtype=np.int32)
            
            # Generate all (t-1)-subsets of the current block
            for S in itertools.combinations(block, t - 1):
                S_set = set(S)
                # For each x not in block:
                # Check if S U {x} is a valid t-subset (it is, if x not in S).
                # And if it is uncovered.
                
                # We can iterate over x not in block.
                # To speed up, we can precompute for each S the list of x?
                # No, S changes.
                
                # Let's just loop x.
                # v=19, block size <= 7.
                # x candidates: 12-19.
                # t-1 subsets: C(7,3)=35.
                # 35 * 19 = 665 operations. Very fast.
                
                for x in range(v):
                    if x in block:
                        continue
                    # S is a tuple. Create the t-subset.
                    # Sort to match key in subset_to_idx
                    new_sub = tuple(sorted(S + (x,)))
                    idx = subset_to_idx.get(new_sub, -1)
                    if idx != -1 and uncovered[idx]:
                        scores[x] += 1
            
            # Find x with max score
            # Mask out points already in block
            mask = np.ones(v, dtype=bool)
            for b in block:
                mask[b] = False
            
            # If all scores are 0, pick any available x (arbitrary)
            # This shouldn't happen if there are uncovered subsets, but just in case.
            available_scores = np.where(mask, scores, -1)
            best_x = np.argmax(available_scores)
            
            block.append(best_x)
        
        # Block is complete. Sort it for consistency (though not strictly required for set logic)
        block.sort()
        blocks.append(block)
        
        # Update uncovered bitset
        # For each t-subset of the block, mark as covered
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            uncovered[idx] = 0
            
    return blocks
