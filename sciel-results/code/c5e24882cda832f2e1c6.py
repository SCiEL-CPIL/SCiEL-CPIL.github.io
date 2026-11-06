"""Rewritten program: search for closest strict underpacking of 1 by unit fractions from [2..48]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

# Denominators are 2..48, zero-based indices 0..46
DENOMS = list(range(2, 49))
N = len(DENOMS)  # 47

def _precompute_reciprocals():
    return [Fraction(1, d) for d in DENOMS]

RECIPS = _precompute_reciprocals()

def _greedy_seed():
    """Greedy: add smallest denominator that keeps sum < 1, then try local swaps."""
    selected = set()
    total = Fraction(0)
    # Greedy by increasing denominator (largest reciprocals first)
    for i in range(N):
        if total + RECIPS[i] < 1:
            selected.add(i)
            total += RECIPS[i]
    return list(selected), total

def _local_improve(indices, deadline):
    """Try to improve the sum by swapping elements or adding/removing single elements."""
    indices = set(indices)
    total = sum((RECIPS[i] for i in indices), Fraction(0))
    best = (total, set(indices))
    
    # Try single swaps: remove one, add one not in set
    removed_candidates = list(indices)
    added_candidates = [i for i in range(N) if i not in indices]
    
    for r in removed_candidates:
        if time.time() > deadline:
            break
        new_total = total - RECIPS[r]
        for a in added_candidates:
            if time.time() > deadline:
                break
            cand_total = new_total + RECIPS[a]
            if cand_total < 1:
                if cand_total > best[0]:
                    best = (cand_total, (indices - {r}) | {a})
    
    # Try adding a single element
    for a in added_candidates:
        if time.time() > deadline:
            break
        cand_total = total + RECIPS[a]
        if cand_total < 1 and cand_total > best[0]:
            best = (cand_total, indices | {a})
    
    # Try removing a single element (if it improves? No, removing decreases sum, which is worse for minimization of deficit)
    # Actually we want to MINIMIZE deficit = 1 - sum, so we want to MAXIMIZE sum while staying < 1.
    # So removing never helps.
    
    return list(best[1]), best[0]

def _beam_search(deadline):
    """Beam search: maintain top-K partial sums, extend with next available denominators."""
    # Start with empty set
    # State: (sum, frozenset_of_indices)
    # We'll use a beam of size BEAM_SIZE
    BEAM_SIZE = 5000
    
    # Initialize beam with empty set
    beam = [(Fraction(0), frozenset())]
    
    # We'll iterate through denominators in order of decreasing reciprocal (increasing denominator index is decreasing reciprocal)
    # Actually, to find good packings, we should consider all subsets. Beam search over subsets is tricky.
    # Alternative: use a priority queue / best-first search with pruning.
    
    # Let's try a different approach: random restarts with local search.
    # Or: meet-in-the-middle? 47 items is too many.
    
    # Let's try simulated annealing or random walk.
    
    # Actually, let's try a more systematic approach:
    # The greedy seed gives a good starting point. Let's do random local search from there.
    
    pass  # Not used, we'll do random search below

def _random_local_search(deadline):
    """Random restarts with local search."""
    best_total = Fraction(0)
    best_indices = []
    
    # Start with greedy seed
    seed_indices, seed_total = _greedy_seed()
    if seed_total > best_total:
        best_total = seed_total
        best_indices = seed_indices
    
    # Try multiple random starts
    import random
    rng = random.Random(42)
    
    max_iterations = 100000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Random start: pick a random subset
        # Bias towards larger subsets (more terms)
        k = rng.randint(5, 20)
        rand_indices = set(rng.sample(range(N), k))
        rand_total = sum((RECIPS[i] for i in rand_indices), Fraction(0))
        
        if rand_total >= 1:
            # Try removing some elements to get below 1
            while rand_total >= 1 and rand_indices:
                # Remove the largest reciprocal (smallest denominator)
                max_i = min(rand_indices)  # smallest denominator = largest reciprocal
                rand_indices.remove(max_i)
                rand_total -= RECIPS[max_i]
        
        # Local search from this start
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices_list = list(rand_indices)
            total = rand_total
            
            # Try single swaps
            removed_candidates = list(rand_indices)
            added_candidates = [i for i in range(N) if i not in rand_indices]
            
            for r in removed_candidates:
                if time.time() > deadline:
                    break
                new_total = total - RECIPS[r]
                for a in added_candidates:
                    if time.time() > deadline:
                        break
                    cand_total = new_total + RECIPS[a]
                    if cand_total < 1 and cand_total > total:
                        rand_indices = (rand_indices - {r}) | {a}
                        total = cand_total
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding a single element
            if not improved:
                for a in added_candidates:
                    if time.time() > deadline:
                        break
                    cand_total = total + RECIPS[a]
                    if cand_total < 1 and cand_total > total:
                        rand_indices = rand_indices | {a}
                        total = cand_total
                        improved = True
        
        if total > best_total:
            best_total = total
            best_indices = list(rand_indices)
    
    return best_indices, best_total

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    # Get greedy seed
    seed_indices, seed_total = _greedy_seed()
    
    # Do random local search
    best_indices, best_total = _random_local_search(deadline)
    
    # Final local improvement
    best_indices, best_total = _local_improve(best_indices, deadline)
    
    # Verify
    assert len(best_indices) > 0, "Must return nonempty list"
    assert len(set(best_indices)) == len(best_indices), "Indices must be distinct"
    assert all(0 <= i < N for i in best_indices), "Indices must be in range"
    total = sum((RECIPS[i] for i in best_indices), Fraction(0))
    assert total < 1, "Sum must be strictly below 1"
    
    return best_indices
# EVOLVE-BLOCK-END
