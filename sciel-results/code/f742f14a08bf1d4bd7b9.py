"""Column-generation / weighted-greedy covering for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

# We'll use a weighted greedy approach with periodic re-weighting (like a simplified
# column generation / dual fitting heuristic).

def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _build_membership(v, k, t):
    """For each point x, precompute the list of t-subsets that contain x.
    But that's too big. Instead, for a partial block B (size < k), the gain of adding x
    is the number of uncovered t-subsets S such that S \ B has size t - len(B) and
    x in S and S subset of B ∪ {x}.
    
    Actually the gain of adding x to block B (|B| = m < k) is:
      count of uncovered t-subsets S where S ⊆ B ∪ {x} and x ∈ S.
    This equals: number of (t - m)-element subsets T of (B ∪ {x}) \ {x} = B such that
    T ∪ {x} ∪ (B \ T) ... wait.
    
    Let me re-derive. Block B has m elements. We add x (not in B). New block B' = B ∪ {x}, size m+1.
    The new t-subsets covered are those t-subsets S of B' that contain x and were not
    already covered (i.e., S not subset of B, which is automatic since x ∉ B, so any S
    containing x is new to this block, but may have been covered by other blocks).
    
    So gain(x) = |{ S ∈ uncovered : |S| = t, x ∈ S, S ⊆ B ∪ {x} }|.
    S ⊆ B ∪ {x} and x ∈ S means S = {x} ∪ T where T ⊆ B and |T| = t - 1.
    So gain(x) = number of (t-1)-subsets T of B such that T ∪ {x} is in uncovered.
    """
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    all_ts = _all_t_subsets(v, t)
    n_ts = len(all_ts)
    
    # uncovered: set of t-subsets not yet covered
    uncovered = set(all_ts)
    blocks = []
    
    # For efficiency, maintain a count of how many times each t-subset is "requested"
    # We'll use a weighted greedy: at each step, choose the block that maximizes
    # sum of weights of uncovered t-subsets it covers.
    
    # Initial weights: uniform
    weights = {ts: 1.0 for ts in all_ts}
    
    # Precompute for each t-subset, the list of points in it (trivial)
    # For pricing: given current weights on uncovered t-subsets, find the k-block
    # that maximizes the total weight of uncovered t-subsets it covers.
    
    # This is a maximum weight k-subset problem. For v=19, k=7, t=4:
    # A block B covers t-subset S iff S ⊆ B.
    # Objective: maximize sum_{S ⊆ B, S uncovered} w[S].
    
    # We can solve this via a branch-and-bound or greedy heuristic.
    # Given the small size (v=19, k=7), we can use a greedy construction for the
    # pricing subproblem: start with empty set, add point that maximizes marginal gain.
    
    def price_block(uncovered_set, w_map):
        """Find a k-block maximizing sum of weights of uncovered t-subsets it covers.
        Uses greedy with lookahead."""
        if not uncovered_set:
            return None, 0.0
        
        best_block = None
        best_gain = -1
        
        # Try multiple random seeds for the greedy pricing
        for seed in range(20):
            random.seed(seed)
            block = set()
            gain = 0.0
            
            for _ in range(k):
                best_x = None
                best_marginal = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Shuffle for diversity
                random.shuffle(candidates)
                
                for x in candidates:
                    # Marginal gain: t-subsets S where x ∈ S, S ⊆ block ∪ {x}, S uncovered
                    # S = {x} ∪ T, T ⊆ block, |T| = t-1
                    marginal = 0.0
                    if len(block) >= t - 1:
                        for T in itertools.combinations(sorted(block), t - 1):
                            S = tuple(sorted(T + (x,)))
                            if S in uncovered_set and S in w_map:
                                marginal += w_map[S]
                    else:
                        # Can't form a t-subset yet with just x and current block
                        # unless t-1 <= len(block)
                        pass
                    
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_x = x
                
                if best_x is None:
                    break
                block.add(best_x)
                gain += best_marginal
            
            if len(block) == k and gain > best_gain:
                best_gain = gain
                best_block = sorted(block)
        
        if best_block is None:
            return None, 0.0
        return best_block, best_gain
    
    # Main loop: weighted greedy with periodic re-weighting
    reweight_interval = 10  # reweight every 10 blocks
    blocks_since_reweight = 0
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Price a new block
        new_block, gain = price_block(uncovered, weights)
        
        if new_block is None or gain <= 0:
            # No beneficial block found; break
            break
        
        # Add the block
        blocks.append(new_block)
        
        # Update uncovered
        for S in itertools.combinations(new_block, t):
            if S in uncovered:
                uncovered.discard(S)
        
        blocks_since_reweight += 1
        
        # Periodic re-weighting: increase weights of uncovered t-subsets
        # to focus on hard-to-cover subsets
        if blocks_since_reweight >= reweight_interval:
            for ts in uncovered:
                weights[ts] *= 2.0
            blocks_since_reweight = 0
    
    # If we timed out or got stuck, do a final greedy pass to cover remaining
    if uncovered and time.time() < deadline - 5.0:
        # Simple greedy for remaining
        while uncovered and time.time() < deadline - 1.0:
            # Pick the uncovered t-subset with highest weight
            first = max(uncovered, key=lambda s: weights.get(s, 1.0))
            block = list(first)
            
            while len(block) < k and time.time() < deadline - 1.0:
                best_x = None
                best_marginal = -1
                for x in range(v):
                    if x in block:
                        continue
                    marginal = 0
                    for T in itertools.combinations(block, t - 1):
                        S = tuple(sorted(T + (x,)))
                        if S in uncovered:
                            marginal += 1
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                # Check if block is new
                if block not in blocks:
                    blocks.append(block)
                    for S in itertools.combinations(block, t):
                        uncovered.discard(S)
            else:
                break
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(key))
    
    return unique_blocks

# EVOLVE-BLOCK-END
