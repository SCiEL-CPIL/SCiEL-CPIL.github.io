import time
import random
import math
import numpy as np
from collections import deque

N_VERT = 12
M = 40
TIME_LIMIT = 75  # seconds, leave 15s margin for 90s wall-clock limit

def _count_pairs(masks):
    """Count comparable pairs in a list of integer masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _count_pairs_np(masks_np):
    """Count comparable pairs using NumPy vectorization."""
    n = len(masks_np)
    a = masks_np[:, np.newaxis]
    b = masks_np[np.newaxis, :]
    ab = a & b
    comparable = (ab == a) | (ab == b)
    # Exclude diagonal
    np.fill_diagonal(comparable, False)
    return int(np.sum(comparable) // 2)

def _delta_for_swap_np(masks_np, i, new_mask):
    """Compute change in comparable-pair count when masks[i] is replaced by new_mask, using NumPy."""
    old = masks_np[i]
    if new_mask == old:
        return 0
    others = np.delete(masks_np, i)
    # Old contributions
    ab_old = others & old
    comp_old = (ab_old == others) | (ab_old == old)
    # New contributions
    ab_new = others & new_mask
    comp_new = (ab_new == others) | (ab_new == new_mask)
    return int(np.sum(comp_new) - np.sum(comp_old))

def _best_replacement_np(masks_np, i, all_masks_np, current_set):
    """Find the best replacement for position i using vectorized evaluation."""
    old = masks_np[i]
    others = np.delete(masks_np, i)
    
    # Precompute old contributions
    ab_old = others & old
    comp_old = (ab_old == others) | (ab_old == old)
    old_contrib = int(np.sum(comp_old))
    
    best_delta = 0
    best_new = None
    
    # Filter out masks already in current set
    available = []
    for mask in all_masks_np:
        if mask not in current_set:
            available.append(mask)
    
    if not available:
        return 0, None
    
    available_np = np.array(available, dtype=np.int64)
    
    # Vectorize: for each candidate, compute comparable count with others
    # Shape: (len(available), len(others))
    ab = available_np[:, np.newaxis] & others[np.newaxis, :]
    comp = (ab == available_np[:, np.newaxis]) | (ab == others[np.newaxis, :])
    new_contribs = comp.sum(axis=1)
    
    deltas = new_contribs - old_contrib
    max_idx = np.argmax(deltas)
    max_delta = int(deltas[max_idx])
    
    if max_delta > 0:
        return max_delta, int(available_np[max_idx])
    return 0, None

def _greedy_init(seed=42):
    """Greedy construction: build a dense comparable structure by adding masks that maximize marginal gain."""
    all_masks = list(range(1 << N_VERT))
    
    # Start with empty set and full set as anchors
    selected = [0, (1 << N_VERT) - 1]
    selected_set = set(selected)
    
    # Greedily add masks that maximize comparable pairs with existing set
    # Evaluate ALL masks to ensure best choice is found
    for _ in range(M - 2):
        best_mask = None
        best_gain = -1
        
        for mask in all_masks:
            if mask in selected_set:
                continue
            # Count how many existing sets are comparable with this mask
            gain = 0
            for existing in selected:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
            selected_set.add(best_mask)
    
    return selected

def _local_search(masks, deadline):
    """Vectorized local search: try all possible replacements for each position using NumPy."""
    current = list(masks)
    current_np = np.array(current, dtype=np.int64)
    current_score = _count_pairs_np(current_np)
    best = list(current)
    best_score = current_score
    
    all_masks_np = np.arange(1 << N_VERT, dtype=np.int64)
    
    while time.time() < deadline:
        improved = False
        current_set = set(current)
        
        for i in range(M):
            old = current[i]
            
            # Find best replacement using vectorized computation
            best_delta, best_new = _best_replacement_np(current_np, i, all_masks_np, current_set)
            
            if best_new is not None and best_delta > 0:
                current[i] = best_new
                current_np[i] = best_new
                current_set.remove(old)
                current_set.add(best_new)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
        
        if not improved:
            break
    
    return best, best_score

def _polish(masks, deadline):
    """Final high-precision polish: try all single-position replacements multiple times until convergence."""
    current = list(masks)
    current_np = np.array(current, dtype=np.int64)
    current_score = _count_pairs_np(current_np)
    best = list(current)
    best_score = current_score
    
    all_masks_np = np.arange(1 << N_VERT, dtype=np.int64)
    
    # Multiple passes to ensure convergence
    while time.time() < deadline:
        improved = False
        current_set = set(current)
        
        # Try each position multiple times in one pass
        for pass_num in range(3):
            for i in range(M):
                old = current[i]
                
                best_delta, best_new = _best_replacement_np(current_np, i, all_masks_np, current_set)
                
                if best_new is not None and best_delta > 0:
                    current[i] = best_new
                    current_np[i] = best_new
                    current_set.remove(old)
                    current_set.add(best_new)
                    current_score += best_delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
        
        if not improved:
            break
    
    return best, best_score

def _build_comparability_graph():
    """Build comparability graph: edge between i,j if masks i and j are comparable."""
    n = 1 << N_VERT
    # Use bitset representation for efficiency
    # comp_graph[i] is a bitset where bit j is set if i and j are comparable
    comp_graph = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = i, j
            if (a & b) == a or (a & b) == b:
                comp_graph[i] |= (1 << j)
                comp_graph[j] |= (1 << i)
    return comp_graph

def _branch_and_bound(comp_graph, target_size, deadline, initial_solution=None):
    """
    Branch and bound to find maximum clique of size target_size in comparability graph.
    Returns list of vertex indices or None if not found.
    """
    n = len(comp_graph)
    best_clique = initial_solution if initial_solution else []
    best_size = len(best_clique)
    
    # Convert to numpy for faster bit operations
    comp_np = np.array(comp_graph, dtype=np.uint64)
    
    def _popcount(x):
        return bin(x).count('1')
    
    def _search(candidates, current_clique, depth):
        nonlocal best_clique, best_size
        
        if time.time() >= deadline:
            return
        
        # Pruning: if even adding all candidates can't beat best, return
        if len(current_clique) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Order candidates by degree in candidate set (descending)
        def _degree_in_candidates(v):
            mask = comp_np[v]
            cand_mask = 0
            for c in candidates:
                cand_mask |= (1 << c)
            return _popcount(mask & cand_mask)
        
        # Sort by degree descending
        candidates_sorted = sorted(candidates, key=_degree_in_candidates, reverse=True)
        
        for v in candidates_sorted:
            if time.time() >= deadline:
                return
            
            new_clique = current_clique + [v]
            if len(new_clique) + len(candidates) <= best_size:
                break
            
            # New candidates must be comparable with v
            new_candidates = [c for c in candidates if c > v and (comp_np[v] >> c) & 1]
            
            _search(new_candidates, new_clique, depth + 1)
            
            candidates = [c for c in candidates if c != v]
            if len(current_clique) + len(candidates) <= best_size:
                break
    
    # Initial candidates: all vertices
    _search(list(range(n)), [], 0)
    
    return best_clique if len(best_clique) == target_size else None

def _construct_from_chain_structure():
    """
    Construct a family with high comparable pairs using chain decomposition.
    The idea: use multiple chains that share elements to maximize comparability.
    """
    all_masks = list(range(1 << N_VERT))
    selected = []
    selected_set = set()
    
    # Strategy: Build several long chains and interleave them
    # Chain 1: empty, {0}, {0,1}, {0,1,2}, ..., full
    # Chain 2: empty, {1}, {1,2}, {1,2,3}, ...
    # etc.
    
    # Better: Use a single long chain plus branches
    # Start with empty and full
    selected.append(0)
    selected_set.add(0)
    selected.append((1 << N_VERT) - 1)
    selected_set.add((1 << N_VERT) - 1)
    
    # Build a main chain: 0, 1, 3, 7, 15, ... (add one bit at a time)
    main_chain = []
    current = 0
    for i in range(N_VERT):
        current |= (1 << i)
        main_chain.append(current)
    
    # Add main chain elements
    for mask in main_chain:
        if mask not in selected_set and len(selected) < M:
            selected.append(mask)
            selected_set.add(mask)
    
    # Now greedily add more elements that are comparable to many existing
    for _ in range(M - len(selected)):
        best_mask = None
        best_gain = -1
        
        for mask in all_masks:
            if mask in selected_set:
                continue
            gain = 0
            for existing in selected:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
            selected_set.add(best_mask)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_masks = None
    best_score = -1
    
    # First: try local search approach (fast, reliable)
    masks_ls = _greedy_init(42)
    result_masks, result_score = _local_search(masks_ls, deadline - 20)
    
    if result_score > best_score:
        best_score = result_score
        best_masks = result_masks
    
    # Polish the local search result
    if time.time() < deadline - 10:
        final_masks, final_score = _polish(best_masks, deadline - 5)
        if final_score > best_score:
            best_score = final_score
            best_masks = final_masks
    
    # Second: try branch-and-bound for potentially better solution
    # This is expensive, so only attempt if we have time
    if time.time() < deadline - 15:
        try:
            comp_graph = _build_comparability_graph()
            
            # Start with current best as initial solution
            initial = [m for m in best_masks] if best_masks else []
            
            bb_result = _branch_and_bound(comp_graph, M, deadline - 5, initial)
            
            if bb_result is not None:
                bb_masks = [comp_graph[i] for i in bb_result]  # This is wrong, need actual masks
                # Actually, bb_result contains vertex indices which ARE the masks
                bb_masks = list(bb_result)
                bb_score = _count_pairs(bb_masks)
                
                if bb_score > best_score:
                    best_score = bb_score
                    best_masks = bb_masks
        except Exception:
            pass
    
    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_VERT) if (mask >> b) & 1])
    
    return result
