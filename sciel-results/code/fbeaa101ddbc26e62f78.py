# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a hybrid heuristic:
    1. Greedy set cover with a randomized restart pool.
    2. Local search to remove redundant blocks and add blocks to fix uncovered sets.
    Time budget is enforced to stay under the 120s limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute the t-subsets it participates in? No, that's too big.
    # Instead, we work with a coverage count array.
    
    best_blocks = None
    best_count = float('inf')
    
    # Strategy: Greedy construction with multiple random seeds/restarts
    # and a local improvement phase.
    
    def greedy_construct(seed):
        """
        Greedy set cover: repeatedly pick the block that covers the most uncovered t-subsets.
        To avoid degeneracy and improve quality, we use a randomized selection among top candidates.
        """
        random.seed(seed)
        
        # uncovered[i] = number of blocks covering t-subset i (we want 0 for uncovered)
        # Actually, for greedy, we track which are uncovered.
        # Using a set of uncovered indices might be slow. Using a boolean array or count array.
        
        # Let's use a list of uncovered t-subsets. But updating it is O(n).
        # Better: maintain a count array `cover_count` of size n_t_subsets.
        # A t-subset is covered if cover_count[i] > 0.
        
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        blocks = []
        
        # Precompute for each point x, the list of t-subsets containing x?
        # That's C(v-1, t-1) per point. Total v * C(v-1, t-1).
        # For v=19, t=4: 19 * C(18,3) = 19 * 816 = 15504. This is small.
        # Let's precompute point_to_subsets: for each point p, list of indices of t-subsets containing p.
        
        point_to_subsets = [[] for _ in range(v)]
        for i, s in enumerate(all_t_subsets):
            for p in s:
                point_to_subsets[p].append(i)
        
        # Also, for a partial block B, to evaluate the gain of adding point x,
        # we need the number of uncovered t-subsets that are contained in B ∪ {x}.
        # A t-subset S is contained in B ∪ {x} if S \ {x} ⊆ B.
        # So for a fixed B and x, the gain is the number of uncovered t-subsets S
        # such that x ∈ S and S \ {x} ⊆ B.
        # S \ {x} is a (t-1)-subset of B.
        
        # Precompute for each (t-1)-subset, the list of t-subsets that contain it?
        # No, we need to go from (t-1)-subset to t-subsets by adding one point.
        
        # Let's precompute: for each (t-1)-subset R, and each point x not in R,
        # the t-subset S = R ∪ {x}. We can map (R, x) to the index of S.
        # But R is a tuple. Let's use a dictionary: key = (tuple(R), x) -> idx.
        # Or, simpler: for each point x, and each (t-1)-subset R of the other points,
        # we can find the t-subset S = R ∪ {x}.
        
        # Alternative approach for gain calculation:
        # For a current block B (list of points), and a candidate point x not in B:
        # Gain = number of uncovered t-subsets S such that S ⊆ B ∪ {x} and x ∈ S.
        # This is equivalent to: S \ {x} ⊆ B.
        # Let R = S \ {x}. R is a (t-1)-subset of B.
        # For each (t-1)-subset R of B, there is exactly one point x such that S = R ∪ {x}
        # is a t-subset? No, S is determined by R and x. But x is fixed in the loop.
        # So for a fixed x, we iterate over all (t-1)-subsets R of B.
        # For each R, S = R ∪ {x}. If S is a valid t-subset (i.e., all elements distinct),
        # then we check if S is uncovered.
        
        # Number of (t-1)-subsets of B is C(|B|, t-1).
        # In the first iteration, |B|=0, so C(0, t-1) = 0 (if t>1). Gain is 0?
        # Wait, if B is empty, we can't form a t-subset by adding one point.
        # The greedy algorithm usually starts by picking an arbitrary point or the one with max degree.
        # Actually, the standard greedy for set cover picks the set (block) with max coverage.
        # But here we are constructing the block point by point.
        
        # Let's stick to the seed program's approach but optimize the gain calculation.
        # The seed program used:
        # gain(x) = sum(1 for rest in itertools.combinations(block, t-1) if tuple(sorted(rest + (x,))) in uncovered)
        
        # We can speed this up by precomputing, for each point x, a set of t-subsets containing x.
        # But the condition is S \ {x} ⊆ B.
        
        # Let's precompute a dictionary: for each (t-1)-subset R (as a tuple), 
        # and for each point x not in R, we have the index of S = R ∪ {x}.
        # Actually, we can just compute the index on the fly using subset_to_idx.
        # The bottleneck is the `in uncovered` check. If `uncovered` is a set, it's O(1).
        # But updating `uncovered` is expensive.
        
        # Let's use the cover_count array.
        # A t-subset is uncovered if cover_count[idx] == 0.
        
        # To find the best x to add to B:
        # For each x not in B:
        #   gain = 0
        #   for R in combinations(B, t-1):
        #       S = tuple(sorted(R + (x,)))
        #       if S in subset_to_idx:
        #           idx = subset_to_idx[S]
        #           if cover_count[idx] == 0:
        #               gain += 1
        #   # Also, if t == 1, we just add x if it covers any uncovered singleton.
        
        # This is O(v * C(|B|, t-1)).
        # |B| goes from 0 to k-1.
        # C(6, 3) = 20. So it's fast.
        
        while True:
            # Check if all are covered
            # Instead of checking all, we can track the number of uncovered.
            # But checking min(cover_count) > 0 is O(n).
            # Let's maintain a variable `uncovered_count`.
            
            # For the first block, B is empty.
            # We need to start B.
            # The seed program started with min(uncovered).
            # Let's pick a random point or the one that appears in the most uncovered t-subsets.
            # For simplicity, let's pick a random point from the first uncovered t-subset.
            
            # Find an uncovered t-subset
            # We can just pick index 0 if it's uncovered, or scan.
            # To be efficient, we can keep a list of uncovered indices.
            # But updating it is hard.
            # Let's just scan for the first uncovered one.
            
            uncovered_idx = -1
            for i in range(n_t_subsets):
                if cover_count[i] == 0:
                    uncovered_idx = i
                    break
            
            if uncovered_idx == -1:
                break # All covered
            
            # Start a new block with one point from this uncovered subset
            # To add randomness, pick a random point from the uncovered subset
            s = all_t_subsets[uncovered_idx]
            start_point = random.choice(s)
            B = [start_point]
            
            # Expand B to size k
            while len(B) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in B:
                        continue
                    gain = 0
                    # Iterate over (t-1)-subsets of B
                    for R in itertools.combinations(B, t-1):
                        # S = R ∪ {x}
                        S = tuple(sorted(R + (x,)))
                        if S in subset_to_idx:
                            idx = subset_to_idx[S]
                            if cover_count[idx] == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    # This shouldn't happen if k <= v
                    break
                
                B.append(best_x)
            
            # Add the block
            B.sort()
            blocks.append(B)
            
            # Update cover_count
            for S in itertools.combinations(B, t):
                idx = subset_to_idx[tuple(S)]
                cover_count[idx] += 1
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return blocks
    
    # Local search: remove redundant blocks
    def local_improve(blocks):
        """
        Try to remove blocks that are redundant (i.e., removing them doesn't uncover any t-subset).
        """
        if not blocks:
            return blocks
        
        # Build cover_count
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for B in blocks:
            for S in itertools.combinations(B, t):
                idx = subset_to_idx[tuple(S)]
                cover_count[idx] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each block
            for i in range(len(blocks)):
                B = blocks[i]
                # Check if any t-subset in B has cover_count == 1
                redundant = True
                for S in itertools.combinations(B, t):
                    idx = subset_to_idx[tuple(S)]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    blocks.pop(i)
                    # Decrement cover_count
                    for S in itertools.combinations(B, t):
                        idx = subset_to_idx[tuple(S)]
                        cover_count[idx] -= 1
                    improved = True
                    break # Restart the loop since indices changed
        
        return blocks
    
    # Main loop: try multiple greedy constructions with different seeds
    seed = 0
    while time.time() < deadline - 10:
        blocks = greedy_construct(seed)
        
        # Local improvement
        blocks = local_improve(blocks)
        
        # Verify coverage (just in case)
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for B in blocks:
            for S in itertools.combinations(B, t):
                idx = subset_to_idx[tuple(S)]
                cover_count[idx] += 1
        
        if np.all(cover_count > 0):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we hit the known lower bound or a very good number, we might stop early?
                # But let's keep searching for a bit.
        
        seed += 1
        
        # Occasionally, try a more aggressive local search or a different heuristic?
        # For now, just restart greedy.
    
    if best_blocks is None:
        # Fallback: return the last constructed blocks even if not verified (should not happen)
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
