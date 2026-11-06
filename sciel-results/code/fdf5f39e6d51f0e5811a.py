import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

N = 20
DEADLINE_FRACTION = 0.85  # safety margin

def _is_k33_free_masks(masks):
    """Check if graph with bitmask adjacency is K3,3-free."""
    for a in range(N):
        ma = masks[a]
        for b in range(a + 1, N):
            common_ab = ma & masks[b]
            if common_ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                common_abc = common_ab & masks[c]
                if common_abc.bit_count() >= 3:
                    return False
    return True

def _edge_count_masks(masks):
    return sum(m.bit_count() for m in masks) // 2

def _can_add_edge_masks(masks, u, v):
    """Check if adding edge (u,v) keeps graph K3,3-free."""
    # Temporarily add the edge
    masks[u] |= (1 << v)
    masks[v] |= (1 << u)
    
    free = _is_k33_free_masks(masks)
    
    # Revert
    masks[u] &= ~(1 << v)
    masks[v] &= ~(1 << u)
    
    return free

def _random_dense_masks(n, p=0.5, seed=None):
    """Generate a random graph with edge probability p using bitmasks."""
    rng = random.Random(seed)
    masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def _greedy_add_masks(masks):
    """Try to add all possible edges greedily."""
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if not (masks[u] & (1 << v)):
                    if _can_add_edge_masks(masks, u, v):
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        improved = True
    return masks

def _edge_swap_search_masks(masks, max_iters=5000, time_limit=100):
    """Local search via edge swaps: remove one edge, add another."""
    start_time = time.time()
    best_masks = masks[:]
    best_count = _edge_count_masks(best_masks)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge to remove
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if masks[i] & (1 << j):
                    edges.append((i, j))
        if not edges:
            break
        u_rm, v_rm = random.choice(edges)
        
        # Try to add a random non-edge
        non_edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if not (masks[i] & (1 << j)):
                    non_edges.append((i, j))
        if not non_edges:
            break
        u_add, v_add = random.choice(non_edges)
        
        # Perform swap
        masks[u_rm] &= ~(1 << v_rm)
        masks[v_rm] &= ~(1 << u_rm)
        masks[u_add] |= (1 << v_add)
        masks[v_add] |= (1 << u_add)
        
        # Check K3,3-free
        if _is_k33_free_masks(masks):
            count = _edge_count_masks(masks)
            if count > best_count:
                best_count = count
                best_masks = masks[:]
        else:
            # Revert
            masks[u_rm] |= (1 << v_rm)
            masks[v_rm] |= (1 << u_rm)
            masks[u_add] &= ~(1 << v_add)
            masks[v_add] &= ~(1 << u_add)
    
    return best_masks, best_count

def construct():
    """Construct a K3,3-free graph on 20 vertices maximizing edges."""
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION  # Safety margin before 150s limit
    
    best_masks = None
    best_count = 0
    
    # Strategy: Multiple restarts with different initializations
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Start with a random dense graph
            masks = _random_dense_masks(N, p=0.5, seed=restart)
        elif restart % 3 == 1:
            # Random init with varying density
            p = 0.3 + 0.4 * (restart % 5) / 4.0
            masks = _random_dense_masks(N, p=p, seed=restart * 7 + 1)
        else:
            # Start from best found so far with perturbation
            if best_masks is not None:
                masks = best_masks[:]
                # Randomly remove some edges
                for _ in range(random.randint(5, 15)):
                    edges = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            if masks[i] & (1 << j):
                                edges.append((i, j))
                    if not edges:
                        break
                    u, v = random.choice(edges)
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
            else:
                masks = _random_dense_masks(N, p=0.4, seed=restart * 13 + 3)
        
        # Check if K3,3-free; if not, try to fix by removing edges
        while not _is_k33_free_masks(masks):
            # Find a K3,3 and remove an edge from it
            # Simple approach: randomly remove edges until K3,3-free
            n_edges = _edge_count_masks(masks)
            if n_edges == 0:
                break
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if masks[i] & (1 << j):
                        edges.append((i, j))
            if not edges:
                break
            random.shuffle(edges)
            removed = False
            for u, v in edges:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                if _is_k33_free_masks(masks):
                    removed = True
                    break
            if not removed:
                break
        
        # Greedy add
        masks = _greedy_add_masks(masks)
        count = _edge_count_masks(masks)
        
        if count > best_count:
            best_count = count
            best_masks = masks[:]
        
        # Edge swap local search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            masks_swapped, count_swapped = _edge_swap_search_masks(
                masks, max_iters=2000, time_limit=min(remaining_time - 2, 30)
            )
            if count_swapped > best_count:
                best_count = count_swapped
                best_masks = masks_swapped[:]
    
    # Final greedy pass on best
    if best_masks is not None and time.time() < deadline - 5:
        best_masks = _greedy_add_masks(best_masks)
        best_count = _edge_count_masks(best_masks)
    
    # Convert to edge list
    if best_masks is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(N):
        m = best_masks[i]
        for j in range(i + 1, N):
            if m & (1 << j):
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
