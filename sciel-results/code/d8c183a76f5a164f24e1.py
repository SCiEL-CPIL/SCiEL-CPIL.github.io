import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # 1. Precomputation
    # Generate all t-subsets (constraints)
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Generate all k-subsets (potential blocks/variables)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map t-subsets to indices for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, find which t-subsets it covers
    # block_tsubs[i] is a list of indices of t-subsets covered by block i
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # For each t-subset, find which blocks cover it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # 2. Branch and Bound Solver
    # State: 
    # - uncovered: set of indices of t-subsets not yet covered
    # - selected: list of selected block indices
    # - lower_bound: greedy estimate of blocks needed to cover 'uncovered'
    
    best_solution = None
    best_size = float('inf')
    
    # Initial Greedy to get a good upper bound
    def greedy_init(uncovered_set):
        """
        Greedily select blocks to cover uncovered_set.
        Returns list of selected blocks.
        """
        selected = []
        current_uncovered = set(uncovered_set)
        
        while current_uncovered:
            if check_time(2.0):
                return None # Timeout, abort this branch
            
            # Find block covering most uncovered t-subsets
            best_block = -1
            best_gain = 0
            
            # Optimization: Only consider blocks that cover at least one uncovered t-subset
            # We can iterate through uncovered t-subsets and their candidates
            # But for speed, if num_blocks is small enough, we might just scan all.
            # num_blocks for C(19,8) is 75582. Scanning all is ~75k ops.
            # num_uncovered can be large.
            # Better: Iterate through uncovered, gather candidates, score them.
            
            candidate_counts = {}
            for ts_idx in current_uncovered:
                for b_idx in tsub_blocks[ts_idx]:
                    candidate_counts[b_idx] = candidate_counts.get(b_idx, 0) + 1
                    # Early exit if we found a perfect cover? Unlikely.
                    
            if not candidate_counts:
                return None # Impossible to cover
                
            # Find max
            for b_idx, count in candidate_counts.items():
                if count > best_gain:
                    best_gain = count
                    best_block = b_idx
                    
            if best_block == -1:
                return None
                
            selected.append(best_block)
            
            # Update uncovered
            for ts_idx in block_tsubs[best_block]:
                if ts_idx in current_uncovered:
                    current_uncovered.remove(ts_idx)
                    
        return selected

    # Run initial greedy on all t-subsets
    initial_uncovered = set(range(num_tsubs))
    initial_greedy = greedy_init(initial_uncovered)
    if initial_greedy:
        best_solution = initial_greedy
        best_size = len(initial_greedy)
        
    # B&B Search
    # We will use a recursive approach but with a stack to manage depth/time.
    # However, pure B&B on 75k variables is hard. 
    # We need a strong pruning heuristic.
    
    # Alternative: Use the greedy result as a start and try to "improve" via B&B on the complement?
    # No, let's stick to standard B&B but with aggressive pruning.
    
    # Actually, for v=19, k=8, t=5, the number of blocks is large.
    # A pure B&B might be too slow.
    # Let's use a Hybrid: 
    # 1. Get a good greedy solution.
    # 2. Use Local Search (like the original) but with a B&B-inspired neighborhood?
    # Or, implement a "Limited Discrepancy Search" or "Branch and Bound" that branches on the most constrained t-subset.
    
    # Let's implement a Branch and Bound that branches on t-subsets.
    # Node: (uncovered_set, selected_blocks)
    # Branching: Pick an uncovered t-subset. Branch on which block covers it.
    
    # To make it efficient:
    # 1. Order blocks for each t-subset by their "potential" (how many other uncovered t-subsets they cover).
    # 2. Prune if len(selected) + lower_bound(uncovered) >= best_size.
    
    # Lower Bound: 
    # LB1: ceil(|uncovered| / max_coverage_per_block)
    # LB2: Greedy Lower Bound (run a fast greedy on uncovered, count steps, but stop early if it exceeds best_size - len(selected))
    
    # Since Greedy is expensive, let's use LB1 and a simple disjoint set LB.
    # LB_disjoint: Find a set of pairwise disjoint t-subsets in uncovered. Each block can cover at most one (since blocks are k-subsets and t-subsets are disjoint? No, blocks can cover multiple disjoint t-subsets if k is large enough).
    # Wait, a k-subset can contain at most floor(k/t) disjoint t-subsets.
    # For k=8, t=5, a block can contain at most 1 disjoint 5-subset.
    # So if we find m pairwise disjoint uncovered t-subsets, we need at least m more blocks.
    
    def get_disjoint_lb(uncovered_set):
        """
        Estimate lower bound based on disjoint t-subsets.
        Greedily pick t-subsets from uncovered that are disjoint from already picked ones.
        """
        if not uncovered_set:
            return 0
            
        # Convert to list for iteration
        # This is expensive if done naively. 
        # Optimization: Use a bitset for uncovered? 
        # For now, simple greedy:
        remaining = list(uncovered_set)
        random.shuffle(remaining) # Randomize to get different LBs? No, deterministic is better for pruning consistency.
        
        picked = []
        # To check disjointness, we need to know the points in the t-subset.
        # tsub_index is ts -> idx. We need idx -> ts.
        # Let's precompute tsub_points: idx -> tuple of points
        
        # Precomputation outside B&B loop
        # tsub_points = [all_tsubs[i] for i in range(num_tsubs)]
        
        # Inside LB:
        available = set(uncovered_set)
        count = 0
        # We want to find a maximal set of disjoint t-subsets.
        # Greedy: Pick one, remove all intersecting ones.
        while available:
            # Pick first (or random)
            ts_idx = next(iter(available))
            count += 1
            # Remove ts_idx and all intersecting ones
            # Intersecting means sharing any point.
            # Precompute for each t-subset idx, the set of t-subset indices that intersect it?
            # That's a huge matrix.
            # Instead, check intersection on the fly.
            # tsub_points[ts_idx] is a tuple of 5 ints.
            # For each other ts_idx2 in available, check if set(tsub_points[ts_idx]) & set(tsub_points[ts_idx2]) is empty.
            # This is O(N^2) in worst case. Too slow.
            
            # Simplified LB: Just ceil(|uncovered| / C(k,t))
            # C(8,5) = 56.
            # LB = ceil(len(uncovered) / 56)
            # This is very weak.
            
            break # Fallback to simple LB for speed in this iteration
            
        return len(uncovered_set) // 56 + (1 if len(uncovered_set) % 56 > 0 else 0)
    
    # Let's refine the strategy. The pure B&B might be too slow.
    # The original WalkSAT was decent. 
    # Let's try a "Greedy + Local Search" but with a different neighborhood: 
    # "Swap" is good. What about "Remove 2, Add 1"? (2-for-1 improvement)
    # Or "Remove 1, Add 2" (to escape local minima)?
    
    # Actually, let's implement a proper B&B with a tight time limit per node.
    
    # Precompute tsub_points for disjoint check if needed
    tsub_points = all_tsubs # all_tsubs[i] is the tuple
    
    def lower_bound(uncovered_set):
        # Simple LB
        return math.ceil(len(uncovered_set) / 56.0)
        
    def branch_and_bound(uncovered_set, selected_blocks, depth=0):
        nonlocal best_solution, best_size
        
        if check_time(5.0):
            return
            
        if not uncovered_set:
            if len(selected_blocks) < best_size:
                best_size = len(selected_blocks)
                best_solution = list(selected_blocks)
            return
            
        # Prune
        if len(selected_blocks) + lower_bound(uncovered_set) >= best_size:
            return
            
        # Branching: Pick an uncovered t-subset with fewest candidates
        # This is a standard heuristic for CSP/SAT
        min_candidates = float('inf')
        target_ts_idx = -1
        
        # To speed up, don't check all uncovered if there are many.
        # Sample a few?
        # For now, check all if small, else sample.
        if len(uncovered_set) < 1000:
            candidates_list = list(uncovered_set)
        else:
            # Random sample
            candidates_list = random.sample(list(uncovered_set), min(1000, len(uncovered_set)))
            
        for ts_idx in candidates_list:
            num_cands = len(tsub_blocks[ts_idx])
            if num_cands < min_candidates:
                min_candidates = num_cands
                target_ts_idx = ts_idx
                
        if target_ts_idx == -1:
            return # No uncovered? Should be handled above.
            
        # Get candidates for target_ts_idx
        # Sort candidates by "gain" (how many other uncovered t-subsets they cover)
        cands = tsub_blocks[target_ts_idx]
        
        # Calculate gain for each candidate
        scored_cands = []
        for b_idx in cands:
            gain = 0
            for ts_idx in block_tsubs[b_idx]:
                if ts_idx in uncovered_set:
                    gain += 1
            scored_cands.append((gain, b_idx))
            
        # Sort by gain descending
        scored_cands.sort(reverse=True)
        
        # Branch
        for gain, b_idx in scored_cands:
            if check_time(5.0):
                return
                
            # New state
            new_selected = selected_blocks + [b_idx]
            
            # Prune before recursing?
            if len(new_selected) + lower_bound(uncovered_set) >= best_size:
                # Since we are adding a block, len increases. 
                # But uncovered decreases. 
                # We need to calculate new_uncovered to get accurate LB.
                # For now, optimistic prune: if len(new_selected) >= best_size, prune.
                if len(new_selected) >= best_size:
                    continue
            
            # Calculate new_uncovered
            # new_uncovered = uncovered_set - set(block_tsubs[b_idx])
            # Using set difference is fast.
            new_uncovered = uncovered_set - set(block_tsubs[b_idx])
            
            # Recurse
            branch_and_bound(new_uncovered, new_selected, depth + 1)
            
            # If we found a solution, best_size might have updated.
            # Pruning subsequent branches in this level
            if len(new_selected) + lower_bound(new_uncovered) >= best_size:
                # Note: lower_bound(new_uncovered) is an estimate. 
                # If this holds, further branches (which will have >= len(new_selected) and similar uncovered) 
                # are likely not better.
                # But strictly, we should continue if there's a chance.
                # However, for performance, we break if we are clearly worse.
                pass

    # Run B&B
    # Start with empty selected, all uncovered
    branch_and_bound(set(range(num_tsubs)), [])
    
    # If B&B didn't find anything better, use the initial greedy
    if best_solution is None:
        best_solution = initial_greedy
        
    # Final reduction (optional, B&B should be minimal)
    # But let's do a quick reduction pass just in case
    if best_solution:
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[block_tsubs[b_idx]] += 1
            
        # Simple reduction
        changed = True
        while changed and not check_time(10.0):
            changed = False
            new_list = []
            for b_idx in best_solution:
                is_essential = False
                for t_idx in block_tsubs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    changed = True
                    for t_idx in block_tsubs[b_idx]:
                        cover_count[t_idx] -= 1
                else:
                    new_list.append(b_idx)
            best_solution = new_list
            
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 8, 5)
    print(f"Found covering with {len(blocks)} blocks")
