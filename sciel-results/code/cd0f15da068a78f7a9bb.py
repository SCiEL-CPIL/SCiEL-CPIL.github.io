"""Program for covering design C(19,8,5) using a dual-guided greedy heuristic."""
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design using a dual-guided greedy approach.
    
    The idea is to maintain weights (dual variables) for each uncovered t-subset.
    When selecting a block, we prefer blocks that cover t-subsets with high dual weights.
    After adding a block, we update the dual weights of newly covered subsets.
    """
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Track which t-subsets are uncovered
    uncovered = [True] * n_tsubsets
    uncovered_count = n_tsubsets
    
    # For each block (k-subset), precompute the list of t-subset indices it covers
    # But C(19,8) = 75582, and each covers C(8,5)=56 t-subsets, so total edges ~ 4.2M
    # We'll compute on the fly to save memory, or precompute if feasible
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # This is C(19,8) * C(8,5) = 75582 * 56 = 4,232,592 entries
    # Store as a dict mapping frozenset(k_subset) -> list of t-subset indices
    # Actually, let's just compute the gain on the fly but use a smarter scoring
    
    blocks = []
    
    # Dual weights: initially 1 for all uncovered
    dual = [1.0] * n_tsubsets
    
    # For efficiency, maintain for each point x, the list of t-subsets containing x
    # And for each pair, etc. But let's try a simpler approach first.
    
    # Precompute: for each t-subset, its index
    # For each potential block candidate, compute the sum of dual weights of uncovered t-subsets it covers
    
    # To make this efficient, we'll use the following approach:
    # 1. Start with the uncovered t-subset that has the highest dual weight
    # 2. Build a block around it by greedily adding points that maximize the weighted coverage
    # 3. Update dual weights after each block
    
    # Precompute for each point, the list of t-subsets that contain it
    point_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubs[x].append(i)
    
    # For faster lookup, also precompute for each pair of points, the t-subsets containing both
    # But that might be too much. Let's try a different approach.
    
    # Alternative: for each t-subset, store it as a frozenset for quick membership tests
    
    while uncovered_count > 0:
        # Find the uncovered t-subset with the highest dual weight
        # To avoid scanning all n_tsubsets each time, we could maintain a priority queue,
        # but for simplicity, let's just find it.
        # Actually, n_tsubsets = C(19,5) = 11628, so scanning is fine.
        
        best_tsub_idx = -1
        best_dual = -1
        for i in range(n_tsubsets):
            if uncovered[i] and dual[i] > best_dual:
                best_dual = dual[i]
                best_tsub_idx = i
        
        if best_tsub_idx < 0:
            break
        
        # Start the block with the points from this t-subset
        block = list(all_tsubsets[best_tsub_idx])
        
        # Now greedily add points to maximize the weighted coverage of uncovered t-subsets
        while len(block) < k:
            block_set = set(block)
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute the gain: sum of dual weights of uncovered t-subsets that contain all of block
                # and x, but are not yet covered
                # A t-subset is covered by block ∪ {x} if it is a subset of block ∪ {x}
                # i.e., it contains at most (t - len(block)) points from outside block
                # and all its points are in block ∪ {x}
                
                # More precisely: a t-subset S is covered by block ∪ {x} iff S ⊆ block ∪ {x}
                # Since |block| = b, |block ∪ {x}| = b+1, and |S| = t,
                # S ⊆ block ∪ {x} means S \ {x} ⊆ block, so S contains at most 1 point not in block,
                # and that point must be x.
                
                # So we need: S \ {x} ⊆ block, i.e., all points of S except possibly x are in block.
                # This means S has at least t - 1 points in block (if x ∈ S) or t points in block (if x ∉ S, but then S ⊆ block, which is impossible since |block| < t typically... wait)
                
                # Actually, if x ∉ S, then S ⊆ block is required, but |S| = t and |block| = b < k.
                # If b >= t, then some t-subsets are already covered by block alone.
                
                # Let's just compute: for each uncovered t-subset, check if it's a subset of block ∪ {x}
                # But that's O(n_tsubsets) per candidate x, and we have v candidates, so O(v * n_tsubsets) per step.
                # n_tsubsets = 11628, v = 19, so ~220K operations per point addition. With k - t = 3 additions, that's ~660K per block.
                # With ~300 blocks, that's ~200M operations. Might be slow in Python.
                
                # Let's optimize: precompute for each block state, which t-subsets are "almost covered"
                # A t-subset is almost covered by block if |S ∩ block| >= t - 1
                # Then adding x covers it iff x ∈ S and S \ {x} ⊆ block, i.e., |S ∩ block| = t - 1 and x ∈ S \ block
                
                # Let's compute the gain more efficiently.
                
                # For candidate x, the t-subsets that become covered are those S where:
                # 1. S is uncovered
                # 2. S ⊆ block ∪ {x}
                # 3. S is not already covered by block (i.e., S ⊄ block)
                
                # Condition 2 and 3 together: S \ block = {x}, so |S ∩ block| = t - 1 and x ∈ S.
                
                # So we need to count uncovered t-subsets S with |S ∩ block| = t - 1 and x ∈ S.
                
                # We can precompute for the current block, for each point x not in block,
                # the list of t-subsets that have exactly t-1 points in block and contain x.
                
                # Let's do it by iterating over t-subsets that intersect block in exactly t-1 points.
                # A t-subset S with |S ∩ block| = t - 1 has exactly one point outside block.
                # So S = (block ∩ S) ∪ {y} where y ∉ block.
                # For a fixed x, we want S such that y = x, i.e., S = (block ∩ S) ∪ {x} where |block ∩ S| = t - 1.
                # So block ∩ S is a (t-1)-subset of block, and S = that (t-1)-subset ∪ {x}.
                
                # So for each (t-1)-subset R of block, the t-subset R ∪ {x} is a candidate.
                # We need to check if R ∪ {x} is a valid t-subset (it is, since |R| = t-1 and x ∉ block ⊇ R).
                # And check if it's uncovered.
                
                gain = 0.0
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    idx = tsub_to_idx.get(S, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += dual[idx]
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point < 0 or best_gain <= 0:
                # No point improves coverage; break or add a random point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_point = random.choice(remaining)
                else:
                    break
            
            block.append(best_point)
        
        # Sort the block
        block.sort()
        blocks.append(block)
        
        # Update uncovered status and dual weights
        block_set = set(block)
        for i in range(n_tsubsets):
            if uncovered[i]:
                ts = all_tsubsets[i]
                if all(x in block_set for x in ts):
                    uncovered[i] = False
                    uncovered_count -= 1
                    # Boost dual weights of other uncovered t-subsets that share points with this one?
                    # Or just leave them. The dual adjustment is implicit in the greedy selection.
        
        # Optional: adjust dual weights to encourage covering harder-to-cover subsets
        # Increase dual for uncovered t-subsets that are "hard" (few blocks can cover them)
        # For simplicity, we'll keep dual at 1.0 for now, but we could increase it for subsets
        # that remain uncovered after many iterations.
    
    return blocks
