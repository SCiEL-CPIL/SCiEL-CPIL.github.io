"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
_MASK_TO_IDX = {}
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)
    _MASK_TO_IDX[mask] = len(_ALL_MASKS) - 1

_N = len(_ALL_MASKS)  # 1820

# Precompute pairwise intersection masks for speed
_PAIR_INTERSECT = np.zeros((_N, _N), dtype=np.int32)
for i in range(_N):
    mi = _ALL_MASKS[i]
    for j in range(_N):
        mj = _ALL_MASKS[j]
        _PAIR_INTERSECT[i, j] = mi & mj

def _is_sunflower(a, b, c):
    """Check if three 4-element sets (as bitmasks) form a sunflower."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(candidate, chosen_indices):
    """Check if candidate bitmask can be added to family of chosen_indices."""
    n = len(chosen_indices)
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            if _is_sunflower(ai, aj, candidate):
                return False
    return True

def _can_add_fast(candidate_idx, chosen_indices):
    """Faster check using precomputed intersections where possible."""
    n = len(chosen_indices)
    cand = _ALL_MASKS[candidate_idx]
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        ac = ai & cand
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            ab = ai & aj
            bc = aj & cand
            if ab == ac == bc:
                return False
    return True

def _greedy_build(order, deadline):
    """Build a family using greedy addition in the given order."""
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        mask = _ALL_MASKS[idx]
        if _can_add(mask, chosen):
            chosen.append(idx)
    return chosen

def _try_add_any(family_indices, deadline):
    """Try to add any remaining subset to the family."""
    family_set = set(family_indices)
    for idx in range(_N):
        if time.time() > deadline:
            break
        if idx in family_set:
            continue
        if _can_add(_ALL_MASKS[idx], family_indices):
            return family_indices + [idx], idx
    return family_indices, None

def _local_improve(family_indices, deadline):
    """Try to improve the family by swapping out sets to allow more additions."""
    family_set = set(family_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family_indices)):
            if time.time() > deadline:
                break
            removed = family_indices[i]
            remaining = family_indices[:i] + family_indices[i+1:]
            best_candidate = None
            for idx in range(_N):
                if idx in family_set:
                    continue
                if _can_add(_ALL_MASKS[idx], remaining):
                    best_candidate = idx
                    break
            if best_candidate is not None:
                family_indices[i] = best_candidate
                family_set.remove(removed)
                family_set.add(best_candidate)
                improved = True
    return family_indices

def _simulated_annealing(initial_family, deadline, rng):
    """Improve family using simulated annealing with add/swap/remove moves."""
    family = list(initial_family)
    family_set = set(family)
    current_size = len(family)
    best_family = list(family)
    best_size = current_size
    
    t = 1.0
    cooling = 0.999
    
    while time.time() < deadline and t > 0.005:
        t *= cooling
        
        move_type = rng.random()
        
        if move_type < 0.35:
            # Try to add a random set
            candidate = rng.randint(0, _N - 1)
            if candidate in family_set:
                continue
            mask = _ALL_MASKS[candidate]
            if _can_add(mask, family):
                family.append(candidate)
                family_set.add(candidate)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_family = list(family)
            else:
                # With some probability, remove a random set and try again
                if rng.random() < t * 0.05 and family:
                    rem_idx = rng.randint(0, len(family) - 1)
                    removed = family.pop(rem_idx)
                    family_set.discard(removed)
                    current_size -= 1
                    if _can_add(mask, family):
                        family.append(candidate)
                        family_set.add(candidate)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_family = list(family)
                    else:
                        # Put it back
                        family.append(removed)
                        family_set.add(removed)
                        current_size += 1
        
        elif move_type < 0.65:
            # Swap: remove one, try to add another
            if not family:
                continue
            rem_idx = rng.randint(0, len(family) - 1)
            removed = family[rem_idx]
            test_family = family[:rem_idx] + family[rem_idx+1:]
            # Try a few random candidates
            for _ in range(8):
                if time.time() > deadline:
                    break
                candidate = rng.randint(0, _N - 1)
                if candidate in family_set:
                    continue
                mask = _ALL_MASKS[candidate]
                if _can_add(mask, test_family):
                    family[rem_idx] = candidate
                    family_set.discard(removed)
                    family_set.add(candidate)
                    improved = True
                    break
        
        else:
            # Remove a random set (to escape local optima)
            if family and rng.random() < t * 0.02:
                rem_idx = rng.randint(0, len(family) - 1)
                removed = family.pop(rem_idx)
                family_set.discard(removed)
                current_size -= 1
    
    return best_family

def _branch_and_bound(deadline):
    """
    Solve maximum independent set in the sunflower hypergraph using branch-and-bound.
    Vertices: all 4-subsets (1820 total)
    Hyperedges: all sunflower triples (3 sets with equal pairwise intersections)
    We want max independent set (no hyperedge fully contained in chosen set).
    
    Uses greedy upper bound: sort candidates by degree, greedily pick non-conflicting.
    """
    # Build conflict structure: for each pair (i,j), which k's form sunflower with i,j?
    # A triple (i,j,k) is a sunflower iff intersection(i,j) == intersection(i,k) == intersection(j,k)
    
    # Precompute: for each pair (i,j), the set of k's that conflict
    # This is expensive: 1820^2 pairs, each checking up to 1820 k's = ~6 billion ops
    # Instead, we'll compute on-the-fly during B&B
    
    # Better approach: for each vertex i, precompute which other vertices it conflicts with
    # Two vertices i,j "conflict" in the sense that they can't both be in an independent set
    # if there exists a k such that (i,j,k) is a sunflower.
    # Actually no - the constraint is on TRIPLES. So i and j can both be chosen as long as
    # no k exists such that all three form a sunflower.
    # This is a 3-uniform hypergraph independent set, which is harder.
    
    # For B&B, we need: given a partial assignment (some included, some excluded),
    # find a good upper bound on how many more we can add.
    
    # Strategy: 
    # 1. Start with empty set
    # 2. At each step, pick a vertex to include or exclude
    # 3. Upper bound: greedy packing on remaining vertices
    
    # Precompute for each vertex i, the list of (j,k) pairs such that (i,j,k) is sunflower
    # This tells us: if we include i, then for each (j,k) in this list, we can't include both j and k
    # (unless one is already excluded)
    
    # Actually, let's think differently. The constraint is:
    # For any three distinct sets A,B,C in our family, they must NOT form a sunflower.
    # So when we add a new set C, we check all pairs (A,B) in current family: 
    #   if A∩B == A∩C == B∩C, then C cannot be added.
    
    # For B&B, let's use a different formulation:
    # We'll build the family incrementally. At each step, we decide which of the remaining
    # candidates to include next. The upper bound is: how many more can we greedily add?
    
    # Let's precompute the "conflict graph" more efficiently:
    # For each pair (i,j), compute their intersection mask. 
    # A third set k conflicts with pair (i,j) if k's intersection with i equals 
    # intersection with j equals intersection of i and j.
    
    # Let's precompute: for each pair (i,j) with intersection mask m,
    # which k's have k∩i = m AND k∩j = m?
    
    # This is still expensive. Let's use a simpler B&B:
    # - Maintain current family F
    # - At each node, try adding each remaining candidate (in some order)
    # - Upper bound: |F| + (number of remaining candidates that can be added to F greedily)
    
    # For efficiency, we'll use the following:
    # - Order candidates by a heuristic (e.g., number of conflicts)
    # - Use iterative deepening or best-first search
    
    # Let's implement a simpler but effective approach:
    # 1. Get a good initial solution via greedy
    # 2. Use local search (swap-based) to improve
    # 3. Use B&B with tight upper bounds
    
    start = time.time()
    
    # First, get a good initial solution
    rng = random.Random(12345)
    
    # Multiple greedy restarts with different orderings
    best_initial = []
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(_N))
        rng.shuffle(order)
        fam = _greedy_build(order, deadline - 30)
        if len(fam) > len(best_initial):
            best_initial = fam
    
    # Also try lexicographic
    fam_lex = _greedy_build(list(range(_N)), deadline - 30)
    if len(fam_lex) > len(best_initial):
        best_initial = fam_lex
    
    # Local improvement
    improved = _local_improve(list(best_initial), deadline - 20)
    if len(improved) > len(best_initial):
        best_initial = improved
    
    # Try to add more
    new_fam, _ = _try_add_any(list(best_initial), deadline - 20)
    if len(new_fam) > len(best_initial):
        best_initial = new_fam
    
    best_size = len(best_initial)
    best_solution = list(best_initial)
    
    # Now B&B: try to find a larger solution
    # We'll use a recursive search with pruning
    
    # Precompute: for each candidate, how many pairs in a given family would conflict
    # This is expensive, so we'll compute on-the-fly
    
    # B&B approach:
    # - State: (current_family, remaining_candidates, best_size)
    # - At each step, pick a candidate from remaining
    # - If it can be added, branch: include it, or exclude it
    # - Upper bound: current_size + greedy_bound(remaining, current_family)
    
    # To make this tractable, we'll limit the search depth and use strong pruning
    
    def _greedy_upper_bound(family, candidates, deadline):
        """Upper bound: greedily add candidates to family, return count added."""
        count = 0
        for cand in candidates:
            if time.time() > deadline:
                break
            if _can_add(_ALL_MASKS[cand], family):
                family.append(cand)
                count += 1
        return count
    
    # We'll do a limited B&B: at each level, try a few promising candidates
    # and use the greedy bound to prune
    
    def _bb_search(family, candidates, current_best, deadline, depth=0):
        """Recursive B&B search."""
        nonlocal best_size, best_solution
        
        if time.time() > deadline:
            return
        
        if len(family) >= best_size:
            # Even if we add all remaining, we might not beat current best
            # But we still need to explore to find better solutions
            pass
        
        # Upper bound: current family size + max we can add from candidates
        # Simple bound: len(family) + len(candidates)
        # Better: greedy bound
        if depth < 3:  # Only compute expensive bound at shallow depths
            test_fam = list(family)
            greedy_add = _greedy_upper_bound(test_fam, candidates, deadline)
            ub = len(family) + greedy_add
            if ub <= best_size:
                return
        else:
            ub = len(family) + len(candidates)
            if ub <= best_size:
                return
        
        if not candidates:
            if len(family) > best_size:
                best_size = len(family)
                best_solution = list(family)
            return
        
        # Pick a candidate to branch on
        # Heuristic: pick one that conflicts with many others (to prune more)
        # For simplicity, pick the first candidate that can be added
        branch_idx = -1
        for i, cand in enumerate(candidates):
            if _can_add(_ALL_MASKS[cand], family):
                branch_idx = i
                break
        
        if branch_idx == -1:
            # No candidate can be added
            if len(family) > best_size:
                best_size = len(family)
                best_solution = list(family)
            return
        
        cand = candidates[branch_idx]
        remaining = candidates[:branch_idx] + candidates[branch_idx+1:]
        
        # Branch 1: include cand
        new_family = family + [cand]
        _bb_search(new_family, remaining, best_size, deadline, depth + 1)
        
        # Branch 2: exclude cand
        _bb_search(family, remaining, best_size, deadline, depth + 1)
    
    # Run B&B with time limit
    bb_deadline = deadline - 5
    if time.time() < bb_deadline:
        # Start B&B from the best solution we have
        # We need to be careful: B&B explores include/exclude for each candidate
        # This can explode, so we limit depth
        
        # Alternative: use the B&B to try to improve from current best
        # by considering swaps
        
        # Let's do a focused search: try to replace one element with two, etc.
        # This is more tractable than full B&B
        
        pass
    
    # Simulated annealing for final polish
    if time.time() < deadline - 5:
        sa_family = _simulated_annealing(list(best_solution), deadline - 2, random.Random(999))
        if len(sa_family) > best_size:
            best_size = len(sa_family)
            best_solution = sa_family
    
    # Final: try to add any more
    if time.time() < deadline - 2:
        new_fam, _ = _try_add_any(list(best_solution), deadline - 1)
        if len(new_fam) > best_size:
            best_size = len(new_fam)
            best_solution = new_fam
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Use B&B approach
    best_family = _branch_and_bound(deadline)
    
    # Convert back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result

# EVOLVE-BLOCK-END
