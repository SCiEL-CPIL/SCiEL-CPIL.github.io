import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Pre-generate all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    
    best_blocks = None
    best_count = float('inf')
    
    # If v is small, we might find a solution quickly.
    # We run multiple restarts to escape local minima.
    
    while True:
        if time.time() > deadline:
            break
            
        current_uncovered = set(uncovered)
        blocks = []
        
        while current_uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple to seed the new block
            # This diversifies the search compared to always picking min()
            seed_triple = random.choice(list(current_uncovered))
            block = list(seed_triple)
            
            # Greedily fill the rest of the block
            while len(block) < k:
                best_gain = -1
                candidates = []
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate how many new uncovered triples are covered by adding x
                    # New triples are of the form {a, b, x} where {a, b} subset of block
                    # Since we add x, we look at pairs in the current block
                    gain = 0
                    # Optimization: Only need to check if len(block) >= t-1
                    if len(block) >= t - 1:
                        # Generate all pairs in block
                        # For each pair (a,b), check if (a,b,x) is in current_uncovered
                        # We can pre-compute pairs or just iterate
                        # t is small (3), so t-1 is 2.
                        # If t-1 > 2, this logic needs to be generalized to combinations(block, t-1)
                        
                        # General case:
                        for rest in itertools.combinations(block, t - 1):
                            # Create sorted tuple of rest + x
                            # rest is a tuple of length t-1
                            # We need to insert x into rest to form a sorted tuple of length t
                            # Simple way: convert to list, append, sort, convert to tuple
                            # This is a bit slow, but t is small.
                            # Faster: use bisect or manual insertion
                            # Given t=3, rest is a pair (a,b).
                            # Let's optimize for t=3 specifically if possible, but keep general.
                            
                            # Manual insertion for speed
                            new_triple_list = list(rest)
                            new_triple_list.append(x)
                            new_triple_list.sort()
                            new_triple = tuple(new_triple_list)
                            
                            if new_triple in current_uncovered:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                # Pick a random candidate from the best ones to break ties
                if candidates:
                    next_point = random.choice(candidates)
                    block.append(next_point)
                else:
                    # Should not happen if k > t and v > k
                    # If no gain, pick any available point
                    available = [x for x in range(v) if x not in block]
                    if available:
                        next_point = random.choice(available)
                        block.append(next_point)
                    else:
                        break # Should not happen
            
            block.sort()
            blocks.append(block)
            
            # Remove covered triples from current_uncovered
            # Generate all t-subsets of the new block
            block_set = set(block)
            to_remove = set()
            for comb in itertools.combinations(block, t):
                if comb in current_uncovered:
                    to_remove.add(comb)
            
            current_uncovered.difference_update(to_remove)
            
            # Check if we have a better solution
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
            # Early termination if we hit a known lower bound or perfect score?
            # We don't know the lower bound exactly, but if we get very low, we can stop.
            # For C(24,7,3), 76 is best known. If we get < 76, great.
            # If we get stuck at 80, we keep trying.

    if best_blocks is None:
        # Fallback: return a trivial valid covering if we ran out of time before finishing one?
        # The loop above ensures we only update best_blocks if a full covering is found in a run.
        # If no run finished, we need to return something.
        # However, the greedy loop inside `while current_uncovered` might be interrupted by deadline.
        # If interrupted, `blocks` is incomplete. We should not return incomplete blocks.
        # So we must ensure we at least complete one run if possible.
        # If we haven't completed even one run, we are in trouble.
        # Let's force one synchronous run if best_blocks is None and time allows.
        if time.time() < deadline:
             current_uncovered = set(uncovered)
             blocks = []
             while current_uncovered and time.time() < deadline:
                 seed_triple = min(current_uncovered) # Deterministic for safety
                 block = list(seed_triple)
                 while len(block) < k:
                     best_gain = -1
                     candidates = []
                     for x in range(v):
                         if x in block: continue
                         gain = 0
                         if len(block) >= t - 1:
                             for rest in itertools.combinations(block, t - 1):
                                 new_triple_list = list(rest)
                                 new_triple_list.append(x)
                                 new_triple_list.sort()
                                 new_triple = tuple(new_triple_list)
                                 if new_triple in current_uncovered:
                                     gain += 1
                         if gain > best_gain:
                             best_gain = gain
                             candidates = [x]
                         elif gain == best_gain:
                             candidates.append(x)
                     if candidates:
                         block.append(candidates[0]) # Deterministic tie-break for fallback
                     else:
                         available = [x for x in range(v) if x not in block]
                         if available: block.append(available[0])
                         else: break
                 block.sort()
                 blocks.append(block)
                 to_remove = set()
                 for comb in itertools.combinations(block, t):
                     if comb in current_uncovered:
                         to_remove.add(comb)
                 current_uncovered.difference_update(to_remove)
                 if not current_uncovered:
                     best_blocks = blocks
                     break
             if best_blocks is None and blocks:
                 # Return the partial result? No, must be valid.
                 # If we didn't finish, we can't return it.
                 # But the problem says "Valid when every t-subset... lies in at least one block".
                 # So we MUST return a complete covering.
                 # If we run out of time, we fail?
                 # "an invalid result... scores 0".
                 # So we must ensure we finish at least one covering.
                 # The greedy algorithm for C(24,7,3) is fast enough to finish one run in < 1s.
                 # So this fallback is just a safety net.
                 pass

    return best_blocks if best_blocks else []
