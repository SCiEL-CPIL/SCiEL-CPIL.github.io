import time
import random
import itertools
import sys
import numpy as np

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def count_size6_violations_full(adj_masks, n):
    """
    Counts size-6 violations.
    """
    count = 0
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            count += 1
    return count

def find_any_violation(adj_masks, n):
    """
    Finds any violation of size >= 6.
    """
    # Size 6
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
            return combo
            
    # Size 7
    for combo in itertools.combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
            return combo
            
    # Size 8+
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
                return combo
    return None

def is_valid_graph(adj_masks, n):
    return find_any_violation(adj_masks, n) is None

def spectral_gap_heuristic(adj_matrix):
    """
    Compute a metric related to the spectral gap.
    We want graphs that are expanders.
    For regular graphs, gap = lambda_1 - lambda_2.
    For non-regular, we can look at the spread of eigenvalues or specifically
    how close the graph is to being a Ramanujan graph (max gap).
    Here we use the difference between the largest eigenvalue and the second largest
    (in magnitude, excluding the trivial one if it's regular-ish).
    Actually, for general graphs, a large spectral gap (lambda_1 - lambda_2)
    where lambda_1 is the largest eigenvalue and lambda_2 is the second largest
    (by value) indicates good mixing/expansion.
    """
    try:
        eigvals = np.linalg.eigvalsh(adj_matrix)
        # eigvals are sorted ascending
        # lambda_1 is eigvals[-1], lambda_2 is eigvals[-2]
        return eigvals[-1] - eigvals[-2]
    except Exception:
        return 0.0

def simulated_annealing_spectral(adj_masks, n, rng, deadline, initial_temp=1.0, cooling=0.99):
    """
    Minimizes size-6 violations using SA.
    Uses delta calculation for speed.
    """
    cur = adj_masks[:]
    cur_energy = count_size6_violations_full(cur, n)
    
    if cur_energy == 0:
        return cur, 0

    T = initial_temp
    min_T = 1e-4
    
    # Precompute lists of others for each pair (u, v) to speed up delta?
    # Or just generate on the fly. C(19,4) is small enough.
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
            
        # Check if edge exists
        edge_exists = bool(cur[u] & (1 << v))
        
        # Calculate delta energy
        # Affected sets: size 6 containing u and v
        others = [x for x in range(n) if x != u and x != v]
        delta = 0
        
        # We need to check regularity of all C(n-2, 4) sets containing u,v
        # Before and After.
        # To avoid full re-calculation, we note:
        # deg_S(x) changes only for u and v.
        # If edge added: deg(u)+1, deg(v)+1.
        # If edge removed: deg(u)-1, deg(v)-1.
        
        for combo_others in itertools.combinations(others, 4):
            S = tuple(sorted([u, v] + list(combo_others)))
            mask = 0
            for x in S:
                mask |= (1 << x)
            
            # Get current degrees in S
            degs = [(cur[x] & mask).bit_count() for x in S]
            
            # Check if regular before
            is_reg_before = all(d == degs[0] for d in degs)
            
            # Apply flip to degrees
            # Find index of u and v in S
            # Since S is sorted, we can find indices
            # But it's easier to just compute new degrees for u and v
            # and check regularity.
            
            # Let's just compute new degrees for u and v
            # Note: degs list is aligned with S
            # We need to map S to degs.
            # S is a tuple of 6 vertices.
            # degs[i] is degree of S[i] in S.
            
            # Find indices of u and v in S
            idx_u = S.index(u)
            idx_v = S.index(v)
            
            new_degs = degs[:]
            if edge_exists:
                new_degs[idx_u] -= 1
                new_degs[idx_v] -= 1
            else:
                new_degs[idx_u] += 1
                new_degs[idx_v] += 1
                
            is_reg_after = all(d == new_degs[0] for d in new_degs)
            
            if is_reg_before and not is_reg_after:
                delta -= 1
            elif not is_reg_before and is_reg_after:
                delta += 1
        
        # Acceptance
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Apply flip
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)
            cur_energy += delta
            
            if cur_energy == 0:
                return cur, 0
        
        T *= cooling
        
    return cur, cur_energy

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_n = 0
    best_edges = []
    
    # Try n=21 first
    target_n = 21
    
    # Phase 1: Generate candidates and rank by spectral gap
    candidates = []
    
    # We need to reserve time for SA.
    # Let's say 30s for generation/ranking, 100s for SA on top candidates.
    gen_deadline = START_TIME + 30
    
    num_candidates = 50
    
    for i in range(num_candidates):
        if time.time() > gen_deadline:
            break
            
        # Random graph with p=0.5 is a good starting point for expansion
        # Or slightly biased
        p = 0.5
        adj_matrix = np.zeros((target_n, target_n))
        adj_masks = [0] * target_n
        
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if rng.random() < p:
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Compute spectral gap
        gap = spectral_gap_heuristic(adj_matrix)
        candidates.append((gap, adj_masks))
        
    # Sort by gap descending (larger gap = better expander = likely fewer regular subgraphs)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: SA on top candidates
    sa_deadline = deadline
    top_k = 5
    
    for i, (gap, masks) in enumerate(candidates[:top_k]):
        if time.time() > sa_deadline:
            break
            
        # Allocate remaining time proportionally?
        # Or just give each a fixed chunk?
        # Let's give each candidate a slice of the remaining time.
        remaining_time = sa_deadline - time.time()
        time_per_candidate = remaining_time / (top_k - i)
        current_deadline = time.time() + time_per_candidate
        
        # Run SA
        refined_masks, energy = simulated_annealing_spectral(masks, target_n, rng, current_deadline)
        
        # Verify full validity (all sizes >= 6)
        if is_valid_graph(refined_masks, target_n):
            edges = get_edges_from_masks(refined_masks, target_n)
            return {'n': target_n, 'edges': edges}
            
        # If SA found 0 size-6 violations, check larger sizes
        # The SA only minimizes size-6. It might have created size-7 violations.
        # So we must do full check.
        
    # If n=21 failed, try n=20
    target_n = 20
    # Re-use some logic or just random restarts for 20?
    # Given 20 is known, we can probably find it quickly.
    # Let's just do a few random restarts for 20 if 21 fails.
    
    for i in range(10):
        if time.time() > deadline:
            break
        p = 0.5
        adj_masks = [0] * target_n
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if rng.random() < p:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
        # Quick local search / SA
        refined_masks, energy = simulated_annealing_spectral(adj_masks, target_n, rng, deadline)
        
        if is_valid_graph(refined_masks, target_n):
            edges = get_edges_from_masks(refined_masks, target_n)
            return {'n': target_n, 'edges': edges}

    # Fallback
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}, edges={len(result['edges'])}")
