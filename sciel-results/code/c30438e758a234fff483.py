import itertools
import time
import random
import sys
import numpy as np
from math import comb

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return selected
    
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        if no_improve_count > 100:
            break
        
        block_to_remove = random.choice(list(best_selected))
        
        uncovered_if_removed = []
        for s in block_covers[block_to_remove]:
            if counts[s] <= 1:
                uncovered_if_removed.append(s)
        
        if not uncovered_if_removed:
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_size -= 1
            no_improve_count = 0
            continue
        
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            no_improve_count += 1
            continue
        
        candidates = sub_to_blocks[target_sub]
        
        best_gain = -1
        best_replacement = None
        for b in candidates:
            if b in best_selected:
                continue
            
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_replacement = b
        
        if best_replacement is not None:
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        min_count = float('inf')
        target_sub = -1
        for s in uncovered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            break
        
        candidates = sub_to_blocks[target_sub]
        
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            gains.append((gain, b))
        
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def _conflict_graph_bnb(block_covers, n_t, sub_to_blocks, initial_solution, deadline, max_nodes=50000):
    """
    Branch-and-bound on the conflict graph of k-blocks.
    
    The conflict graph has vertices = blocks in the initial solution.
    An edge between blocks u and v exists if selecting both is "wasteful":
    specifically, if there exists a t-subset s such that s is covered by BOTH u and v,
    but removing one of them would still leave s covered by some OTHER block in the solution.
    
    Actually, a better model: we want to find a minimum subset of blocks that still covers all t-subsets.
    We use the conflict graph where two blocks are adjacent if they are "redundant together" —
    i.e., their intersection of covered t-subsets is non-empty AND for every t-subset in their
    intersection, there's another block covering it.
    
    For the BnB: we branch on which blocks to keep/remove. The lower bound comes from
    greedy coloring of the "essential coverage" structure.
    """
    n_blocks_in_sol = len(initial_solution)
    if n_blocks_in_sol < 3:
        return initial_solution
    
    sol_list = list(initial_solution)
    
    # Build coverage counts for the initial solution
    counts = [0] * n_t
    for i in sol_list:
        for s in block_covers[i]:
            counts[s] += 1
    
    # Precompute for each block in the solution: which t-subsets it uniquely covers
    unique_cover = [set() for _ in range(n_blocks_in_sol)]
    for idx, b in enumerate(sol_list):
        for s in block_covers[b]:
            if counts[s] == 1:
                unique_cover[idx].add(s)
    
    # Conflict: block i and block j conflict if removing i would uncover some t-subset
    # that j also covers (i.e., j could replace i for those t-subsets)
    # For BnB, we model: we need to select a subset of blocks such that all t-subsets are covered.
    
    # Better approach: use the t-subset coverage structure directly.
    # Lower bound: for each t-subset, it must be covered by at least one selected block.
    # Greedy coloring lower bound on the "set cover" problem.
    
    best_selected = set(sol_list)
    best_size = n_blocks_in_sol
    node_count = [0]
    
    def _compute_lb(uncovered_tsubs, remaining_blocks):
        """Compute a lower bound on additional blocks needed."""
        if not uncovered_tsubs:
            return 0
        
        # Simple LB: each block can cover at most max_cov of the uncovered t-subsets
        max_cov = 0
        for b in remaining_blocks:
            cov = 0
            for s in block_covers[b]:
                if s in uncovered_tsubs:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
        
        if max_cov == 0:
            return float('inf')
        
        return (len(uncovered_tsubs) + max_cov - 1) // max_cov
    
    def _bnb(uncovered_tsubs, remaining_blocks, selected_so_far, depth):
        """Branch and bound to find a better covering."""
        if not _deadline_ok(deadline) or node_count[0] >= max_nodes:
            return None
        
        node_count[0] += 1
        
        if not uncovered_tsubs:
            return selected_so_far
        
        if not remaining_blocks:
            return None
        
        # Lower bound
        lb = _compute_lb(uncovered_tsubs, remaining_blocks)
        if len(selected_so_far) + lb >= best_size:
            return None
        
        # Pick the most constrained t-subset
        min_count = float('inf')
        target = -1
        for s in uncovered_tsubs:
            c = len(sub_to_blocks[s])
            # Only count blocks in remaining_blocks
            actual_c = 0
            for b in remaining_blocks:
                if s in block_covers[b]:
                    actual_c += 1
            if actual_c < min_count:
                min_count = actual_c
                target = s
                if actual_c == 0:
                    return None  # Infeasible
                if actual_c == 1:
                    break
        
        if target == -1:
            return None
        
        # Get candidates that cover target
        candidates = [b for b in remaining_blocks if target in block_covers[b]]
        
        if not candidates:
            return None
        
        # Sort by coverage of uncovered_tsubs (descending)
        candidates.sort(key=lambda b: sum(1 for s in block_covers[b] if s in uncovered_tsubs), reverse=True)
        
        # Branch: try each candidate
        for b in candidates:
            if not _deadline_ok(deadline) or node_count[0] >= max_nodes:
                return None
            
            new_uncovered = set(uncovered_tsubs)
            for s in block_covers[b]:
                new_uncovered.discard(s)
            
            new_remaining = [x for x in remaining_blocks if x != b]
            new_selected = selected_so_far + [b]
            
            result = _bnb(new_uncovered, new_remaining, new_selected, depth + 1)
            
            if result is not None and len(result) < best_size:
                return result
        
        return None
    
    # Start BnB from the initial solution
    # We want to find if we can cover all t-subsets with fewer than best_size blocks
    # Start with all blocks as remaining, and try to build a solution
    
    # Actually, let's approach it differently: start with the full solution,
    # and try to remove blocks one by one, using BnB to verify feasibility
    
    # Simpler: use the BnB to find a minimum cover from scratch, guided by the initial solution
    # But that's expensive. Instead, let's do a "reduction" BnB:
    
    # For each block in the solution, check if we can remove it and still cover everything
    # with at most best_size - 1 blocks total.
    
    # This is essentially what _local_search does, but let's use the coloring LB.
    
    # Alternative structural approach: use the conflict graph for a different purpose.
    # Build a graph where nodes are t-subsets, and edges connect t-subsets that are
    # covered by the same block. Then a clique in this graph = t-subsets all covered by one block.
    # The minimum number of blocks needed >= chromatic number of the "conflict" graph of t-subsets.
    
    # Let me implement a different BnB: branch on t-subsets, not blocks.
    
    def _bnb_tsubsets(uncovered, selected_blocks, deadline):
        """BnB branching on t-subsets."""
        if not _deadline_ok(deadline) or node_count[0] >= max_nodes:
            return None
        
        node_count[0] += 1
        
        if not uncovered:
            return selected_blocks
        
        # LB: each block covers at most C(k,t) t-subsets
        # Better LB: greedy coloring
        max_cov = 0
        for b in range(len(block_covers)):
            cov = 0
            for s in block_covers[b]:
                if s in uncovered:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
        
        if max_cov == 0:
            return None
        
        lb = (len(uncovered) + max_cov - 1) // max_cov
        if len(selected_blocks) + lb >= best_size:
            return None
        
        # Pick most constrained t-subset
        min_count = float('inf')
        target = -1
        for s in uncovered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target = s
                if c == 1:
                    break
        
        if target == -1:
            return None
        
        candidates = sub_to_blocks[target]
        
        # Sort by coverage
        candidates.sort(key=lambda b: sum(1 for s in block_covers[b] if s in uncovered), reverse=True)
        
        for b in candidates:
            if not _deadline_ok(deadline) or node_count[0] >= max_nodes:
                return None
            
            new_uncovered = set(uncovered)
            for s in block_covers[b]:
                new_uncovered.discard(s)
            
            result = _bnb_tsubsets(new_uncovered, selected_blocks + [b], deadline)
            
            if result is not None and len(result) < best_size:
                return result
        
        return None
    
    # Try BnB from scratch with a tight bound
    # Only try if we have enough time
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        result = _bnb_tsubsets(set(range(n_t)), [], deadline)
        if result is not None and len(result) < best_size:
            return result
    
    return initial_solution

def _conflict_coloring_lb(block_covers, n_t, sub_to_blocks, selected, deadline):
    """
    Compute a lower bound on the number of blocks needed using a coloring-based approach.
    
    Idea: Build a graph where nodes are t-subsets in the current uncovered set.
    Two t-subsets are adjacent if they CANNOT be covered by the same block.
    The chromatic number of this graph is a lower bound on the number of blocks needed.
    """
    if not selected:
        return 0
    
    # Count coverage
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    # Find uncovered or singly-covered t-subsets that constrain the solution
    # Actually, for a lower bound on the TOTAL solution, we consider all t-subsets.
    
    # Simpler LB: partition number. Each block covers C(k,t) t-subsets.
    # So LB = ceil(n_t / C(k,t))
    from math import comb
    max_per_block = comb(8, 5) if (8, 5) == (k, t) else comb(k, t)
    return (n_t + max_per_block - 1) // max_per_block

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 12
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                  randomize=(restart > 0), top_k=15)
        
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
                
    # 4. Additional refinement passes on the best solution
    if best_selected is not None and _deadline_ok(deadline):
        for _ in range(3):
            if not _deadline_ok(deadline):
                break
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=150)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
