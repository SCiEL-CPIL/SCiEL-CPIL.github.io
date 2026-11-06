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
# An edge is a 3-mask.
# We need a mapping: edge_mask -> list of indices of 6-subsets that contain it
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

def is_linear(edges, new_edge):
    """Check if new_edge is linear with all existing edges."""
    for e in edges:
        if (e & new_edge) != 0:
            # Intersection size > 1
            # e & new_edge has 2 or 3 bits
            # We can check popcount > 1
            # Since e and new_edge are 3-bit masks, 
            # if they share 2 bits, (e & new_edge) has 2 bits.
            # if they share 3 bits, (e & new_edge) has 3 bits (and e == new_edge).
            # We assume new_edge is not in edges.
            if POP_COUNTS[e & new_edge] > 1:
                return False
    return True

# Precompute popcounts for 16-bit masks
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

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
            
    uncovered = 0
    for c in covered:
        if c == 0:
            uncovered += 1
    return uncovered

def local_search_six(edges, time_limit):
    """
    Local search to minimize the number of uncovered 6-subsets.
    Returns (best_edges, best_uncovered_count, best_alpha)
    """
    current_edges = edges[:]
    current_uncovered = count_uncovered_six(current_edges)
    best_edges = current_edges[:]
    best_uncovered = current_uncovered
    best_alpha = calc_alpha(current_edges)
    
    if best_uncovered == 0:
        return best_edges, best_uncovered, best_alpha

    # We want to minimize uncovered 6-sets.
    # Moves:
    # 1. Add an edge (if linear)
    # 2. Remove an edge
    # 3. Swap an edge (remove one, add another)
    
    # To speed up, we can precompute the change in uncovered count for each potential move?
    # Adding edge e: 
    #   It covers a set of 6-subsets.
    #   Some of these might already be covered by other edges.
    #   The reduction in uncovered count is the number of 6-subsets covered by e 
    #   that are currently uncovered.
    
    # We can maintain the `covered` array to quickly compute this.
    covered = [0] * NUM_SIX
    for e in current_edges:
        for idx in EDGE_TO_SIX[e]:
            covered[idx] += 1
            
    current_uncovered = sum(1 for c in covered if c == 0)
    
    while time.time() < time_limit:
        improved = False
        
        # 1. Try adding an edge
        # Iterate through all possible triples
        # To make it faster, we can prioritize edges that cover many uncovered 6-sets
        # But computing this for all edges is expensive.
        # Let's just iterate and check.
        
        # Optimization: Only consider edges that are linear.
        # We can precompute linear compatibility?
        # For now, just check on the fly.
        
        # Shuffle triples to avoid bias
        triples = ALL_TRIPLES[:]
        random.shuffle(triples)
        
        best_add_delta = 0
        best_add_edge = None
        
        for e in triples:
            if e in current_edges:
                continue
            if not is_linear(current_edges, e):
                continue
            
            # Calculate delta in uncovered count
            # Delta = - (number of 6-subsets covered by e that are currently uncovered)
            gain = 0
            for idx in EDGE_TO_SIX[e]:
                if covered[idx] == 0:
                    gain += 1
            
            if gain > best_add_delta:
                best_add_delta = gain
                best_add_edge = e
        
        if best_add_edge is not None and best_add_delta > 0:
            # Apply the move
            current_edges.append(best_add_edge)
            for idx in EDGE_TO_SIX[best_add_edge]:
                covered[idx] += 1
            current_uncovered -= best_add_delta
            improved = True
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_edges = current_edges[:]
                # Don't calc alpha every time, it's expensive.
                # But we need to know if we reached 0.
                if best_uncovered == 0:
                    best_alpha = calc_alpha(best_edges)
                    return best_edges, best_uncovered, best_alpha
            continue # Restart loop to re-evaluate all moves
        
        # 2. Try swapping an edge
        # Remove e_old, add e_new
        # This is more complex.
        # Let's try a simpler move: Remove an edge that "wastes" coverage?
        # Or just random restarts if stuck.
        
        if not improved:
            # Try removing an edge?
            # Removing an edge increases uncovered count, so it's usually bad.
            # Unless it allows adding 2 new edges that cover more?
            # That's a 2-swap.
            
            # Let's try a 2-swap: Remove 2, Add 2.
            # This is O(E^2 * T^2) which is too slow.
            
            # Instead, let's try a "kick":
            # Remove a random edge, then try to add the best edge.
            # This might allow escaping local optima.
            
            if len(current_edges) > 0:
                # Remove a random edge
                rem_idx = random.randint(0, len(current_edges) - 1)
                e_rem = current_edges[rem_idx]
                current_edges.pop(rem_idx)
                
                for idx in EDGE_TO_SIX[e_rem]:
                    covered[idx] -= 1
                
                # Try to add the best edge
                best_add_delta = 0
                best_add_edge = None
                triples = ALL_TRIPLES[:]
                random.shuffle(triples)
                
                for e in triples:
                    if e in current_edges:
                        continue
                    if not is_linear(current_edges, e):
                        continue
                    
                    gain = 0
                    for idx in EDGE_TO_SIX[e]:
                        if covered[idx] == 0:
                            gain += 1
                    
                    if gain > best_add_delta:
                        best_add_delta = gain
                        best_add_edge = e
                
                if best_add_edge is not None and best_add_delta > 0:
                    current_edges.append(best_add_edge)
                    for idx in EDGE_TO_SIX[best_add_edge]:
                        covered[idx] += 1
                    current_uncovered -= best_add_delta
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_edges = current_edges[:]
                        if best_uncovered == 0:
                            best_alpha = calc_alpha(best_edges)
                            return best_edges, best_uncovered, best_alpha
                else:
                    # If we couldn't add a good edge, maybe revert the removal?
                    # Or keep the removal if it didn't make things worse?
                    # Actually, if we removed an edge, uncovered count increased.
                    # If we didn't add an edge, we are worse off.
                    # Revert.
                    current_edges.append(e_rem)
                    for idx in EDGE_TO_SIX[e_rem]:
                        covered[idx] += 1
                    # current_uncovered goes back up by the gain we lost?
                    # We need to recalculate or track carefully.
                    # Easier to just restart local search from best_edges if stuck.
                    
            # If stuck, restart from best_edges
            current_edges = best_edges[:]
            covered = [0] * NUM_SIX
            for e in current_edges:
                for idx in EDGE_TO_SIX[e]:
                    covered[idx] += 1
            current_uncovered = sum(1 for c in covered if c == 0)
            
            # Random perturbation: shuffle and try again?
            # To avoid infinite loop, we can just break and do a new random restart in main loop.
            break

    # Calculate final alpha for best_edges
    best_alpha = calc_alpha(best_edges)
    return best_edges, best_uncovered, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = []
    best_alpha = N
    best_uncovered = float('inf')
    
    # Initial random packing
    def random_greedy_packing():
        edges = []
        available = ALL_TRIPLES[:]
        random.shuffle(available)
        for t in available:
            if is_linear(edges, t):
                edges.append(t)
        return edges

    # Try multiple restarts
    while time.time() < deadline:
        # Generate a random packing
        candidate_edges = random_greedy_packing()
        
        # Calculate initial uncovered 6-sets
        # If we have time, do local search
        if time.time() < deadline - 10:
            improved_edges, improved_uncovered, improved_alpha = local_search_six(candidate_edges, deadline - 5)
            
            if improved_alpha < best_alpha or (improved_alpha == best_alpha and improved_uncovered < best_uncovered):
                best_alpha = improved_alpha
                best_edges = improved_edges[:]
                best_uncovered = improved_uncovered
                
                if best_alpha == 5:
                    # We found the theoretical minimum
                    break
        else:
            # Just calculate alpha for the random packing
            cand_alpha = calc_alpha(candidate_edges)
            if cand_alpha < best_alpha:
                best_alpha = cand_alpha
                best_edges = candidate_edges[:]
                # Calculate uncovered for tracking
                best_uncovered = count_uncovered_six(best_edges)

    # Convert best_masks back to list of triples [a, b, c]
    result = []
    for mask in best_edges:
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
