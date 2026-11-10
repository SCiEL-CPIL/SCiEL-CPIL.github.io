import itertools
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(v, k, t):
    """
    Constructs a covering design using a constraint-based backtracking search 
    with a bottleneck heuristic that maximizes the minimum remaining degree 
    of uncovered t-subsets.
    """
    start_time = time.time()
    time_limit = 115  # Leave some buffer for evaluation overhead
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map each block to the indices of t-subsets it covers
    block_t_indices = {}
    # Map each t-subset index to the list of blocks that cover it
    t_subset_to_blocks = defaultdict(list)
    
    for b in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_subset_to_blocks[idx].append(b)
        block_t_indices[b] = covered_indices
        
    # Initialize degrees: number of blocks covering each t-subset
    # Initially, all blocks are available
    degrees = [0] * num_t_subsets
    for idx in range(num_t_subsets):
        degrees[idx] = len(t_subset_to_blocks[idx])
        
    # State:
    # uncovered: set of indices of t-subsets not yet covered
    # available_blocks: set of blocks not yet used
    # current_blocks: list of blocks selected so far
    # best_size: size of the best solution found
    # best_solution: the blocks in the best solution
    
    uncovered = set(range(num_t_subsets))
    available_blocks = set(all_blocks)
    current_blocks = []
    best_size = float('inf')
    best_solution = []
    
    # Lower bound helper: 
    # Minimum blocks needed to cover 'uncovered' subsets.
    # Each block covers at most C(k, t) subsets.
    max_cover = len(next(iter(block_t_indices.values()))) if block_t_indices else 0
    
    def lower_bound():
        if not uncovered:
            return 0
        # Weak bound
        return (len(uncovered) + max_cover - 1) // max_cover
        
    def search():
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_blocks) + lower_bound() >= best_size:
            return
            
        if not uncovered:
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = list(current_blocks)
            return
            
        # 1. Find the bottleneck t-subset
        # We look for the uncovered subset with the minimum degree.
        min_deg = float('inf')
        bottleneck_idx = -1
        
        for idx in uncovered:
            if degrees[idx] < min_deg:
                min_deg = degrees[idx]
                bottleneck_idx = idx
                
        if min_deg == 0:
            return # Dead end
            
        # 2. Get candidates: blocks covering the bottleneck
        candidates = []
        for b in t_subset_to_blocks[bottleneck_idx]:
            if b in available_blocks:
                # Calculate heuristic score
                # We want to maximize the minimum remaining degree of subsets covered by b
                min_rem_deg = float('inf')
                cov_count = 0
                
                for s_idx in block_t_indices[b]:
                    if s_idx in uncovered:
                        cov_count += 1
                        # If we pick b, degrees[s_idx] decreases by 1
                        rem_deg = degrees[s_idx] - 1
                        if rem_deg < min_rem_deg:
                            min_rem_deg = rem_deg
                            
                candidates.append((min_rem_deg, cov_count, b))
                
        if not candidates:
            return
            
        # Sort candidates:
        # Primary: max min_rem_deg (preserve flexibility)
        # Secondary: max cov_count (greedy coverage)
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Limit branching to top 5 to control explosion
        for min_rem, cov, b in candidates[:5]:
            # Apply block b
            current_blocks.append(b)
            available_blocks.remove(b)
            
            # Update degrees and uncovered
            newly_covered = []
            for s_idx in block_t_indices[b]:
                if s_idx in uncovered:
                    degrees[s_idx] -= 1
                    uncovered.remove(s_idx)
                    newly_covered.append(s_idx)
                    
            search()
            
            # Undo
            current_blocks.pop()
            available_blocks.add(b)
            for s_idx in newly_covered:
                degrees[s_idx] += 1
                uncovered.add(s_idx)
                
    # Initial Greedy to get a decent upper bound quickly
    unc_greedy = set(range(num_t_subsets))
    avail_greedy = set(all_blocks)
    greedy_blocks = []
    degs_greedy = [len(t_subset_to_blocks[i]) for i in range(num_t_subsets)]
    
    while unc_greedy:
        min_d = float('inf')
        bot = -1
        for idx in unc_greedy:
            if degs_greedy[idx] < min_d:
                min_d = degs_greedy[idx]
                bot = idx
        if min_d == 0:
            break
        best_b = None
        best_score = -1
        for b in t_subset_to_blocks[bot]:
            if b in avail_greedy:
                # Simple score: max coverage
                cov = 0
                for s in block_t_indices[b]:
                    if s in unc_greedy:
                        cov += 1
                if cov > best_score:
                    best_score = cov
                    best_b = b
        if best_b is None:
            break
        greedy_blocks.append(best_b)
        avail_greedy.remove(best_b)
        for s in block_t_indices[best_b]:
            if s in unc_greedy:
                degs_greedy[s] -= 1
                unc_greedy.remove(s)
                
    if greedy_blocks:
        best_size = len(greedy_blocks)
        best_solution = list(greedy_blocks)
        
    search()
    
    return [list(b) for b in best_solution]
