import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = list(U)
    n = len(U)
    if n == 0:
        return 0.0
    max_val = max(U)
    if max_val == 0:
        return 1.0
    # Compute sumset
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute difference set
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_denom = math.log(2 * max_val + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom


def _try_exhaustive(n, max_elem, deadline):
    """Try exhaustive search for small n"""
    best = None
    best_score = -1
    # Generate all combinations of n-1 elements from 1..max_elem (0 is always included)
    candidates = list(range(1, max_elem + 1))
    
    def _rec(start, chosen, current_U):
        nonlocal best, best_score
        if time.time() > deadline - 2:
            return
        if len(chosen) == n - 1:
            U = [0] + chosen
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best = list(U)
            return
        for i in range(start, len(candidates)):
            if time.time() > deadline - 2:
                return
            chosen.append(candidates[i])
            current_U.append(candidates[i])
            _rec(i + 1, chosen, current_U)
            chosen.pop()
            current_U.pop()
    
    _rec(0, [], [])
    return best, best_score


def _local_search(initial_U, deadline, max_iter=5000):
    """Local search by adding/removing/adjusting elements"""
    U = sorted(list(initial_U))
    current_score = _compute_score(U)
    best_U = list(U)
    best_score = current_score
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Generate neighbors
        neighbors = []
        max_elem = max(U) if U else 0
        # Try adding a new element
        for v in range(1, min(max_elem + 50, 200)):
            if v not in U:
                neighbors.append(U + [v])
        # Try removing an element (not 0)
        for i in range(len(U)):
            if U[i] != 0:
                neighbors.append(U[:i] + U[i+1:])
        # Try adjusting an element
        for i in range(len(U)):
            if U[i] != 0:
                for delta in [-2, -1, 1, 2, 5]:
                    new_v = U[i] + delta
                    if new_v > 0 and new_v not in U:
                        new_U = list(U)
                        new_U[i] = new_v
                        new_U.sort()
                        neighbors.append(new_U)
        
        if not neighbors:
            break
        
        best_neighbor = None
        best_neighbor_score = current_score
        
        for nb in neighbors:
            if time.time() > deadline - 5:
                break
            nb_score = _compute_score(nb)
            if nb_score > best_neighbor_score + 1e-12:
                best_neighbor_score = nb_score
                best_neighbor = nb
        
        if best_neighbor is not None:
            U = sorted(best_neighbor)
            current_score = best_neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(U)
        else:
            # Stuck - try a random restart with perturbation
            if random.random() < 0.5:
                # Perturb
                U_perturbed = list(U)
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(U_perturbed) - 1)
                    if U_perturbed[idx] != 0:
                        U_perturbed[idx] = max(1, U_perturbed[idx] + random.randint(-5, 5))
                U_perturbed = sorted(set(U_perturbed))
                U = U_perturbed
                current_score = _compute_score(U)
            else:
                break
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Start with a reasonable initial set
    # Based on known constructions, try geometric-like sequences
    initial_sets = [
        [0, 1, 3],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 4, 16, 64, 256],
        [0, 1, 3, 6, 10, 15, 21],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 1, 4, 9, 16, 25, 36],
    ]
    
    best_overall = [0, 1, 3]
    best_overall_score = _compute_score(best_overall)
    
    # First, try exhaustive search for small sizes
    for n in [4, 5, 6]:
        if time.time() > deadline - 60:
            break
        result, score = _try_exhaustive(n, 30, deadline)
        if result is not None and score > best_overall_score:
            best_overall_score = score
            best_overall = result
    
    # Local search from multiple initial points
    random.seed(42)
    for init in initial_sets:
        if time.time() > deadline - 20:
            break
        result, score = _local_search(init, deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = result
    
    # Also try some random initial sets
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        size = random.randint(4, 12)
        U_rand = [0]
        for _ in range(size - 1):
            U_rand.append(random.randint(1, 100))
        U_rand = sorted(set(U_rand))
        result, score = _local_search(U_rand, deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = result
    
    # Final polish: try to improve the best solution
    if time.time() < deadline - 5:
        result, score = _local_search(best_overall, deadline, max_iter=2000)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = result
    
    return sorted(best_overall)
# EVOLVE-BLOCK-END
