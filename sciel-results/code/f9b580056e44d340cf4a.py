import time
import random
from itertools import combinations

N = 16
ALL_MASK = (1 << N) - 1

# Precompute all 6-subsets as bitmasks
# There are C(16, 6) = 8008 such subsets
SIX_SUBSETS = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            for l in range(k+1, N):
                for m in range(l+1, N):
                    for n in range(m+1, N):
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << n)
                        SIX_SUBSETS.append(mask)

NUM_SIX = len(SIX_SUBSETS)

# Precompute for each possible triple (edge), which 6-subsets it is contained in
EDGE_TO_SIX = {}
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            edge_mask = (1 << i) | (1 << j) | (1 << k)
            contained_in = []
            for idx, six_mask in enumerate(SIX_SUBSETS):
                if (edge_mask & six_mask) == edge_mask:
                    contained_in.append(idx)
            EDGE_TO_SIX[edge_mask] = contained_in

# Precompute all valid triples
ALL_TRIPLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            ALL_TRIPLES.append((1 << a) | (1 << b) | (1 << c))

# Precompute popcounts for 16-bit masks
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

def is_linear(edges_set, new_edge):
    """Check if new_edge is linear with all existing edges (edges_set is a set of masks)."""
    for e in edges_set:
        if (e & new_edge) != 0:
            if POP_COUNTS[e & new_edge] > 1:
                return False
    return True

def calc_alpha(edges_masks):
    """Calculate independence number."""
    if not edges_masks:
        return N
    
    bad = [False] * (1 << N)
    
    for e in edges_masks:
        comp = ALL_MASK ^ e
        sub = comp
        while True:
            bad[e | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    max_alpha = 0
    for i in range(1 << N):
        if not bad[i]:
            pc = POP_COUNTS[i]
            if pc > max_alpha:
                max_alpha = pc
                if max_alpha == N:
                    break
    return max_alpha

def count_uncovered_six(edges_masks):
    """Count how many 6-subsets are NOT covered by any edge."""
    covered = [0] * NUM_SIX
    for e in edges_masks:
        for idx in EDGE_TO_SIX[e]:
            covered[idx] += 1
    uncovered = sum(1 for c in covered if c == 0)
    return uncovered

def local_search_six(edges, time_limit):
    """
    Local search to minimize the number of uncovered 6-subsets.
    Uses a set for fast membership checks.
    Returns (best_edges_list, best_uncovered_count, best_alpha)
    """
    current_edges_set = set(edges)
    current_edges_list = list(current_edges_set)
    
    # Build covered array
    covered = [0] * NUM_SIX
    for e in current_edges_set:
        for idx in EDGE_TO_SIX[e]:
            covered[idx] += 1
    current_uncovered = sum(1 for c in covered if c == 0)
    
    best_edges_list = current_edges_list[:]
    best_uncovered = current_uncovered
    best_alpha = calc_alpha(current_edges_list)
    
    if best_uncovered == 0:
        return best_edges_list, best_uncovered, best_alpha

    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < time_limit:
        improved = False
        
        # Try adding the best edge
        best_add_delta = 0
        best_add_edge = None
        
        for e in ALL_TRIPLES:
            if e in current_edges_set:
                continue
            if not is_linear(current_edges_set, e):
                continue
            
            gain = 0
            for idx in EDGE_TO_SIX[e]:
                if covered[idx] == 0:
                    gain += 1
            
            if gain > best_add_delta:
                best_add_delta = gain
                best_add_edge = e
        
        if best_add_edge is not None and best_add_delta > 0:
            current_edges_set.add(best_add_edge)
            current_edges_list.append(best_add_edge)
            for idx in EDGE_TO_SIX[best_add_edge]:
                covered[idx] += 1
            current_uncovered -= best_add_delta
            improved = True
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_edges_list = current_edges_list[:]
                no_improve_count = 0
                if best_uncovered == 0:
                    best_alpha = calc_alpha(best_edges_list)
                    return best_edges_list, best_uncovered, best_alpha
            continue
        
        # No add move available. Try a swap: remove one edge, add another.
        swap_improved = False
        if len(current_edges_set) > 0:
            # Try each edge as a removal candidate
            edges_list = list(current_edges_set)
            random.shuffle(edges_list)
            
            for e_rem in edges_list:
                if time.time() >= time_limit:
                    break
                
                # Temporarily remove e_rem
                current_edges_set.discard(e_rem)
                rem_idx = current_edges_list.index(e_rem)
                current_edges_list.pop(rem_idx)
                
                for idx in EDGE_TO_SIX[e_rem]:
                    covered[idx] -= 1
                
                # Find best edge to add
                best_add_delta = 0
                best_add_edge = None
                
                for e_new in ALL_TRIPLES:
                    if e_new in current_edges_set:
                        continue
                    if not is_linear(current_edges_set, e_new):
                        continue
                    
                    gain = 0
                    for idx in EDGE_TO_SIX[e_new]:
                        if covered[idx] == 0:
                            gain += 1
                    
                    if gain > best_add_delta:
                        best_add_delta = gain
                        best_add_edge = e_new
                
                if best_add_edge is not None and best_add_delta > 0:
                    # Accept the swap
                    current_edges_set.add(best_add_edge)
                    current_edges_list.append(best_add_edge)
                    for idx in EDGE_TO_SIX[best_add_edge]:
                        covered[idx] += 1
                    current_uncovered -= best_add_delta
                    swap_improved = True
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_edges_list = current_edges_list[:]
                        no_improve_count = 0
                        if best_uncovered == 0:
                            best_alpha = calc_alpha(best_edges_list)
                            return best_edges_list, best_uncovered, best_alpha
                    break
                else:
                    # Revert the removal
                    current_edges_set.add(e_rem)
                    current_edges_list.append(e_rem)
                    for idx in EDGE_TO_SIX[e_rem]:
                        covered[idx] += 1
        
        if not improved and not swap_improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        # If we made a swap but it didn't improve, we're still at the same state
        # (since we only accept improving swaps). So no_improve_count stays.

    best_alpha = calc_alpha(best_edges_list)
    return best_edges_list, best_uncovered, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = []
    best_alpha = N
    best_uncovered = float('inf')
    
    def random_greedy_packing():
        edges_set = set()
        edges_list = []
        available = ALL_TRIPLES[:]
        random.shuffle(available)
        for t in available:
            if is_linear(edges_set, t):
                edges_set.add(t)
                edges_list.append(t)
        return edges_list

    # Try multiple restarts
    while time.time() < deadline:
        candidate_edges = random_greedy_packing()
        
        if time.time() < deadline - 15:
            improved_edges, improved_uncovered, improved_alpha = local_search_six(candidate_edges, deadline - 10)
            
            if improved_alpha < best_alpha or (improved_alpha == best_alpha and improved_uncovered < best_uncovered):
                best_alpha = improved_alpha
                best_edges = improved_edges[:]
                best_uncovered = improved_uncovered
                
                if best_alpha <= 5:
                    break
        else:
            cand_alpha = calc_alpha(candidate_edges)
            if cand_alpha < best_alpha:
                best_alpha = cand_alpha
                best_edges = candidate_edges[:]
                best_uncovered = count_uncovered_six(best_edges)

    # Final polish: try to reduce alpha further by adding/removing edges
    if best_edges and time.time() < deadline - 5:
        polish_set = set(best_edges)
        polish_list = best_edges[:]
        
        # Try adding edges that might reduce alpha
        for e in ALL_TRIPLES:
            if time.time() >= deadline - 5:
                break
            if e in polish_set:
                continue
            if not is_linear(polish_set, e):
                continue
            polish_set.add(e)
            polish_list.append(e)
            new_alpha = calc_alpha(polish_list)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_edges = polish_list[:]
                best_uncovered = count_uncovered_six(best_edges)
            else:
                polish_set.discard(e)
                polish_list.pop()

    result = []
    for mask in best_edges:
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
