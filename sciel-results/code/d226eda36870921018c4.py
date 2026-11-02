import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    
    # U[x] is a set of subsets S in uncovered such that x is in S
    U = [set() for _ in range(v)]
    for S in all_subsets:
        for x in S:
            U[x].add(S)
    
    blocks = []
    
    # Parameters for multi-candidate greedy
    num_candidates = 20
    top_m = 5 # Pick from top M candidates
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random pivot subset
        pivot = random.choice(list(uncovered))
        
        best_block = None
        best_gain = -1
        
        # Generate num_candidates candidates
        for _ in range(num_candidates):
            block = set(pivot)
            
            # Complete the block greedily with randomness
            while len(block) < k:
                gains = {}
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: number of S in U[x] such that S \ {x} is subset of block
                    # Note: S is a tuple of t elements.
                    # We need S \ {x} to be a subset of block.
                    # Since S has size t, and x is in S, S \ {x} has size t-1.
                    # We can check if all elements of S except x are in block.
                    
                    cnt = 0
                    for S in U[x]:
                        # Optimization: if S is not in uncovered, skip? 
                        # U[x] is kept in sync with uncovered, so all S in U[x] are uncovered.
                        # Check if S \ {x} is subset of block
                        # S is a tuple.
                        # We can precompute S_minus_x? No, x varies.
                        # Just check:
                        # if all(y in block for y in S if y != x)
                        
                        # Faster check:
                        # Since t is small (4), we can do:
                        # Create a set of S? No, S is tuple.
                        # Let's just iterate.
                        
                        # Actually, we can precompute a bitmask for each subset?
                        # v=19, t=4.
                        # Bitmask is efficient.
                        pass
                    
                    # Let's use a simpler check for now, t=4 is small
                    # S is a tuple of 4 ints.
                    # We need to check if S \ {x} is in block.
                    # This is equivalent to: S is a subset of block U {x}
                    
                    # To speed this up, we can precompute for each S in U[x] the other 3 elements.
                    # But U[x] changes.
                    
                    # Let's stick to the loop, it's likely fast enough.
                    cnt = 0
                    for S in U[x]:
                        # Check if S \ {x} is subset of block
                        # S is a tuple.
                        # We can convert block to a set for O(1) lookup.
                        # But block is a set already.
                        
                        # Optimization:
                        # If len(block) < t - 1, it's impossible for S \ {x} to be subset of block?
                        # No, S \ {x} has size t-1.
                        # If len(block) < t-1, gain is 0.
                        if len(block) < t - 1:
                            break # gain is 0 for all x? No, just this x.
                        # Actually if len(block) < t-1, no S \ {x} (size t-1) can be subset of block.
                        # So gain is 0.
                        
                        # Let's refine:
                        # Only calculate if len(block) >= t - 1.
                        
                        # For now, just do the check.
                        # To avoid repeated "in block" checks for each element of S,
                        # we can check if S is subset of block U {x}.
                        
                        # Since t=4, S has 4 elements.
                        # If x is in S, the other 3 must be in block.
                        
                        # Let's just do:
                        # is_subset = True
                        # for y in S:
                        #   if y != x and y not in block:
                        #     is_subset = False
                        #     break
                        # if is_subset: cnt += 1
                        
                        # This is 4 checks per S.
                        # 840 S * 4 = 3360 checks per x.
                        # 19 x * 3360 = 63840 checks per step.
                        # 3 steps * 20 candidates * 150 blocks = 57 million checks.
                        # Python can do ~10-50 million simple ops/sec.
                        # This might be tight.
                        
                        # Let's try to optimize.
                        # Precompute a bitmask for each subset S?
                        # And for block?
                        
                        # Let's try the bitmask approach.
                        pass
                    gains[x] = cnt
                if not gains:
                    break
                # Pick from top M
                sorted_gains = sorted(gains.items(), key=lambda item: item[1], reverse=True)
                top_candidates = [x for x, g in sorted_gains[:top_m]]
                # Add some randomness
                if random.random() < 0.5:
                    next_x = max(gains, key=gains.get)
                else:
                    next_x = random.choice(top_candidates)
                block.add(next_x)
            
            # Evaluate final block
            # Gain is number of S in uncovered such that S is subset of block
            # We can calculate this by iterating U[x] for x in block?
            # Or just iterate uncovered?
            # Iterating uncovered (size ~4000) and checking subset is fast.
            
            current_gain = 0
            block_set = block # It is a set
            for S in uncovered:
                # Check if S is subset of block_set
                # S is tuple.
                # all(y in block_set for y in S)
                if all(y in block_set for y in S):
                    current_gain += 1
            
            if current_gain > best_gain:
                best_gain = current_gain
                best_block = block
        
        if best_block is None:
            # Fallback: just take pivot and add random points
            best_block = set(pivot)
            while len(best_block) < k:
                best_block.add(random.randint(0, v-1))
        
        best_block = sorted(best_block)
        blocks.append(best_block)
        
        # Update uncovered and U
        # Find all S in uncovered that are subset of best_block
        to_remove = []
        for S in uncovered:
            if all(y in best_block for y in S):
                to_remove.append(S)
        
        for S in to_remove:
            uncovered.remove(S)
            for x in S:
                U[x].discard(S)
                
    return blocks
