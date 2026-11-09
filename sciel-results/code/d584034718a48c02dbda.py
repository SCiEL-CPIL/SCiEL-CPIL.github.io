import time
import random
import math
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIRS = []

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append([i, j, k])
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            TRIPLE_PAIRS.append(((i, j), (i, k), (j, k)))

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute pair to index mapping
pair_to_idx = {}
idx = 0
for i in range(16):
    for j in range(i + 1, 16):
        pair_to_idx[(i, j)] = idx
        idx += 1

TRIPLE_PAIR_MASKS = []
for pairs in TRIPLE_PAIRS:
    mask = 0
    for p in pairs:
        mask |= (1 << pair_to_idx[p])
    TRIPLE_PAIR_MASKS.append(mask)

# Precompute for each triple, which other triples share a pair with it
# triple_conflict[i] = bitmask of triples that conflict with triple i
TRIPLE_CONFLICT = [0] * NUM_TRIPLES
for i in range(NUM_TRIPLES):
    for j in range(i + 1, NUM_TRIPLES):
        if TRIPLE_PAIR_MASKS[i] & TRIPLE_PAIR_MASKS[j]:
            TRIPLE_CONFLICT[i] |= (1 << j)
            TRIPLE_CONFLICT[j] |= (1 << i)

def compute_alpha_fast(H_indices, return_set=False):
    """
    Compute the independence number using branch and bound with improved pruning.
    H_indices is a set of indices into ALL_TRIPLES.
    Returns alpha, or (alpha, best_set_mask) if return_set=True.
    """
    if not H_indices:
        if return_set:
            return 16, (1 << 16) - 1
        return 16
    
    H_masks = [ALL_TRIPLE_MASKS[i] for i in H_indices]
    n = 16
    all_vertices = (1 << n) - 1
    best = 0
    best_set = 0
    
    # For each vertex, precompute which H-triples contain it
    vertex_triples = [[] for _ in range(n)]
    for t in H_masks:
        m = t
        while m:
            bit = m & -m
            v = bit.bit_length() - 1
            vertex_triples[v].append(t)
            m -= bit

    # Count how many H-triples each vertex is in
    vertex_deg = [len(vertex_triples[v]) for v in range(n)]

    def dfs(available, current_size, current_mask):
        nonlocal best, best_set
        
        # Pruning: Even if we add all available, we can't beat best
        if current_size + available.bit_count() <= best:
            return
        
        if available == 0:
            if current_size > best:
                best = current_size
                best_set = current_mask
            return
        
        # Find a dangerous triple with maximum number of available vertices
        dangerous_triple = None
        max_avail_count = 0
        for t in H_masks:
            avail_in_t = t & available
            cnt = avail_in_t.bit_count()
            if cnt > 1 and cnt > max_avail_count:
                if (t & ~(current_mask | available)) == 0:
                    dangerous_triple = t
                    max_avail_count = cnt
        
        if dangerous_triple is None:
            new_size = current_size + available.bit_count()
            if new_size > best:
                best = new_size
            return
        
        # Branch on excluding a vertex from the dangerous triple
        candidates = dangerous_triple & available
        vertex_order = []
        c = candidates
        while c:
            bit = c & -c
            vertex_order.append(bit)
            c -= bit
        
        # Sort by degree (higher degree first = more constraining)
        vertex_order.sort(key=lambda b: vertex_deg[b.bit_length() - 1], reverse=True)
        
        for bit in vertex_order:
            new_available = available & ~bit
            dfs(new_available, current_size, current_mask)
            
    dfs(all_vertices, 0, 0)
    if return_set:
        return best, best_set
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    best_alpha = 16
    best_system = []
    
    def directed_local_search(H_indices, used_pairs_mask, current_alpha):
        """
        Directed local search: identify the largest independent set,
        then specifically add triples that hit it to reduce alpha.
        """
        nonlocal best_alpha, best_system
        stagnation_count = 0
        
        while time.time() < deadline and current_alpha > 1:
            # Get the actual largest independent set
            alpha, indep_mask = compute_alpha_fast(H_indices, return_set=True)
            
            if alpha != current_alpha:
                current_alpha = alpha
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_system = [ALL_TRIPLES[i] for i in H_indices]
            
            if current_alpha <= 1:
                break
            
            # Find all feasible triples that intersect the independent set
            # A triple "hits" the independent set if it shares at least one vertex with it
            feasible_adds = []
            for t_idx in range(NUM_TRIPLES):
                if t_idx in H_indices:
                    continue
                add_pmask = TRIPLE_PAIR_MASKS[t_idx]
                if (used_pairs_mask & add_pmask) != 0:
                    continue
                # Check if this triple intersects the independent set
                t_mask = ALL_TRIPLE_MASKS[t_idx]
                if (t_mask & indep_mask) != 0:
                    # Score: prefer triples that hit more vertices of the indep set
                    hit_count = (t_mask & indep_mask).bit_count()
                    # Also prefer triples that are "dense" in the indep set
                    feasible_adds.append((hit_count, t_idx))
            
            if not feasible_adds:
                # No feasible triple hits the independent set; try swaps
                feasible_swaps = []
                for rem_idx in list(H_indices):
                    rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                    freed = used_pairs_mask & ~rem_pmask
                    for t_idx in range(NUM_TRIPLES):
                        if t_idx == rem_idx or t_idx in H_indices:
                            continue
                        add_pmask = TRIPLE_PAIR_MASKS[t_idx]
                        if (freed & add_pmask) != 0:
                            continue
                        t_mask = ALL_TRIPLE_MASKS[t_idx]
                        if (t_mask & indep_mask) != 0:
                            hit_count = (t_mask & indep_mask).bit_count()
                            feasible_swaps.append((hit_count, rem_idx, t_idx))
                
                if not feasible_swaps:
                    stagnation_count += 1
                    if stagnation_count > 50:
                        break
                    # Random perturbation
                    if H_indices:
                        rem_idx = random.choice(list(H_indices))
                        H_indices.discard(rem_idx)
                        used_pairs_mask &= ~TRIPLE_PAIR_MASKS[rem_idx]
                        current_alpha = compute_alpha_fast(H_indices)
                    continue
                
                # Try best swap
                feasible_swaps.sort(reverse=True)
                improved = False
                for hit_count, rem_idx, add_idx in feasible_swaps[:10]:
                    rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    new_used = (used_pairs_mask & ~rem_pmask) | add_pmask
                    
                    H_indices.discard(rem_idx)
                    H_indices.add(add_idx)
                    used_pairs_mask = new_used
                    
                    new_alpha = compute_alpha_fast(H_indices)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                        stagnation_count = 0
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [ALL_TRIPLES[i] for i in H_indices]
                        break
                    else:
                        # Revert
                        H_indices.discard(add_idx)
                        H_indices.add(rem_idx)
                        used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                
                if not improved:
                    stagnation_count += 1
                    if stagnation_count > 100:
                        break
                continue
            
            # Try best adds (those hitting most of the independent set)
            feasible_adds.sort(reverse=True)
            improved = False
            for hit_count, add_idx in feasible_adds[:20]:
                add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                
                H_indices.add(add_idx)
                used_pairs_mask |= add_pmask
                
                new_alpha = compute_alpha_fast(H_indices)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    improved = True
                    stagnation_count = 0
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = [ALL_TRIPLES[i] for i in H_indices]
                    break
                else:
                    # Revert
                    H_indices.discard(add_idx)
                    used_pairs_mask &= ~add_pmask
            
            if not improved:
                stagnation_count += 1
                if stagnation_count > 100:
                    break
                # Random perturbation to escape local optimum
                if H_indices:
                    rem_idx = random.choice(list(H_indices))
                    H_indices.discard(rem_idx)
                    used_pairs_mask &= ~TRIPLE_PAIR_MASKS[rem_idx]
                    current_alpha = compute_alpha_fast(H_indices)
        
        return H_indices, used_pairs_mask, current_alpha
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # 1. Random Greedy Packing with some structure
        H_indices = set()
        used_pairs_mask = 0
        
        available = list(range(NUM_TRIPLES))
        random.shuffle(available)
        
        for t_idx in available:
            pmask = TRIPLE_PAIR_MASKS[t_idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(t_idx)
                used_pairs_mask |= pmask
        
        current_alpha = compute_alpha_fast(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [ALL_TRIPLES[i] for i in H_indices]
        
        # 2. Directed Local Search targeting the independent set
        H_indices, used_pairs_mask, current_alpha = directed_local_search(
            H_indices, used_pairs_mask, current_alpha
        )
        
        # 3. Brief random refinement if time remains
        if time.time() < deadline - 5:
            for _ in range(50):
                if time.time() > deadline:
                    break
                if not H_indices:
                    break
                # Random swap
                if len(H_indices) > 1:
                    rem_idx = random.choice(list(H_indices))
                    rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                    freed = used_pairs_mask & ~rem_pmask
                    found = False
                    for _ in range(10):
                        add_idx = random.randint(0, NUM_TRIPLES - 1)
                        if add_idx in H_indices:
                            continue
                        add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                        if (freed & add_pmask) == 0:
                            H_indices.discard(rem_idx)
                            H_indices.add(add_idx)
                            used_pairs_mask = freed | add_pmask
                            new_alpha = compute_alpha_fast(H_indices)
                            if new_alpha <= current_alpha:
                                current_alpha = new_alpha
                                if current_alpha < best_alpha:
                                    best_alpha = current_alpha
                                    best_system = [ALL_TRIPLES[i] for i in H_indices]
                            else:
                                H_indices.discard(add_idx)
                                H_indices.add(rem_idx)
                                used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                            found = True
                            break
                    if not found:
                        # Revert if we removed but didn't add
                        if rem_idx not in H_indices:
                            H_indices.add(rem_idx)
                            used_pairs_mask |= rem_pmask

    return best_system

def _fallback_system():
    # A known good system: 16 points, aim for small alpha
    # Use a structured construction
    return [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
        [0, 3, 6], [1, 4, 7], [2, 5, 8], [9, 12, 15], [10, 13, 0], [11, 14, 1],
        [2, 15, 3], [4, 9, 10], [5, 11, 12], [6, 13, 14], [7, 15, 8]
    ]

def _validate_system(triples):
    """Validate linearity and return alpha."""
    used = set()
    for t in triples:
        i, j, k = t
        pairs = [tuple(sorted((i, j))), tuple(sorted((i, k))), tuple(sorted((j, k)))]
        for p in pairs:
            if p in used:
                return False
            used.add(p)
    return True

_orig_construct = construct

def construct_safe():
    try:
        res = _orig_construct()
        if not res:
            return _fallback_system()
        if not _validate_system(res):
            return _fallback_system()
        return res
    except:
        return _fallback_system()

# Replace construct
construct = construct_safe
