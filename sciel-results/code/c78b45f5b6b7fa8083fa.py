import itertools
import random
import time

def construct(v, k, t):
    # Parameters for C(19, 7, 4)
    # v=19, k=7, t=4
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    
    # We need to cover all_t_subsets.
    # Greedy Set Cover approach:
    # 1. uncovered = set(all_t_subsets)
    # 2. blocks = []
    # 3. while uncovered:
    #    a. Pick a random S from uncovered (or the one with fewest candidates? All have same structure initially).
    #       Actually, picking the S with the fewest *available* blocks that cover it *and* other things might be good.
    #       But standard greedy: Pick the block B that covers the most uncovered elements.
    #       Constraint: B must cover at least one uncovered element.
    #       To be efficient, we pick an uncovered S, then consider only blocks containing S.
    #       Among those, pick the one with max gain.
    
    best_blocks = None
    best_count = float('inf')
    
    deadline = time.time() + 110 # Safety margin
    
    # Precompute for each t-subset, the list of k-subsets containing it?
    # That's 3876 * 455 = ~1.7M tuples. Might be slow to build initially.
    # Let's generate on the fly.
    
    def get_k_subsets_containing(s):
        # s is a tuple of t elements
        remaining = [x for x in points if x not in s]
        # We need k-t more elements from remaining
        for comb in itertools.combinations(remaining, k - t):
            yield tuple(sorted(s + comb))
            
    # To speed up, we can precompute the 4-subsets for each 7-subset?
    # No, let's just check membership.
    
    # Randomized Greedy with Restarts
    num_restarts = 100 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset S to anchor the block selection
            # This introduces randomness to escape local minima
            S = random.choice(list(uncovered))
            
            best_block = None
            max_gain = -1
            
            # Iterate through all k-subsets containing S
            for B in get_k_subsets_containing(S):
                # Calculate gain: number of uncovered t-subsets in B
                gain = 0
                # Generate all t-subsets of B
                # B is a tuple of k elements
                for sub in itertools.combinations(B, t):
                    if sub in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_block = B
            
            if best_block is None:
                # Should not happen if uncovered is not empty
                break
                
            blocks.append(list(best_block))
            
            # Remove all t-subsets of best_block from uncovered
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
                
        if uncovered:
            # Incomplete coverage due to timeout, discard this result
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
            
            # If we hit a very good count, we might want to try local search
            # But for now, just keep finding better greedy solutions
            
    if best_blocks is None:
        # Fallback to a simple deterministic greedy if random failed
        # Or just return an empty list if timeout hit immediately?
        # The problem guarantees a solution exists.
        # If we ran out of time, we should return the best found so far.
        # If best_blocks is None, it means we didn't complete a single run.
        # We should probably run a deterministic one first or ensure at least one completes.
        
        # Let's add a deterministic run if best is None
        if time.time() < deadline:
             uncovered = set(all_t_subsets)
             blocks = []
             while uncovered:
                 if time.time() > deadline: break
                 S = min(uncovered) # Deterministic choice
                 best_block = None
                 max_gain = -1
                 for B in get_k_subsets_containing(S):
                     gain = sum(1 for sub in itertools.combinations(B, t) if sub in uncovered)
                     if gain > max_gain:
                         max_gain = gain
                         best_block = B
                 if best_block is None: break
                 blocks.append(list(best_block))
                 for sub in itertools.combinations(best_block, t):
                     uncovered.discard(sub)
             if not uncovered:
                 best_blocks = blocks
                 best_count = len(blocks)

    return best_blocks
