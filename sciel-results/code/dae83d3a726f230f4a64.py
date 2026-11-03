"""Optimized search for finite set U maximizing sumset/difference-set exponent."""
import time
import random
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def _score(U):
    """Fast score computation using numpy or sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # For small N, Python sets are often faster due to lower overhead
    if n <= 60:
        sums = set()
        diffs = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                sums.add(ui + U[j])
        for i in range(n):
            ui = U[i]
            for j in range(n):
                diffs.add(ui - U[j])
        n_sums = len(sums)
        n_diffs = len(diffs)
    else:
        a = np.array(U, dtype=np.int64)
        sums = a[:, None] + a[None, :]
        n_sums = len(np.unique(sums))
        diffs = a[:, None] - a[None, :]
        n_diffs = len(np.unique(diffs))
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def _has_distinct_diffs(U):
    """Check if all pairwise differences are distinct (maximizes |U-U|)."""
    n = len(U)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _generate_sidon_like(n, max_val, seed=42):
    """Generate a set with distinct pairwise differences using greedy."""
    rng = random.Random(seed)
    U = [0]
    remaining = list(range(1, max_val + 1))
    rng.shuffle(remaining)
    for _ in range(n - 1):
        best = None
        best_score = -1
        for v in remaining[:min(50, len(remaining))]:
            candidate = U + [v]
            candidate.sort()
            if _has_distinct_diffs(candidate):
                # Prefer smaller values to keep max(U) small
                score = -v
                if score > best_score:
                    best_score = score
                    best = v
        if best is not None:
            U.append(best)
            remaining.remove(best)
            U.sort()
        else:
            break
    return sorted(U) if len(U) >= 2 else [0, 1]

def _generate_random_distinct(n, max_val, seed=42):
    """Generate random set, then try to fix collisions."""
    rng = random.Random(seed)
    U = [0] + sorted(rng.sample(range(1, max_val + 1), min(n - 1, max_val)))
    U.sort()
    # Try to fix collisions by shifting
    for _ in range(200):
        if _has_distinct_diffs(U):
            break
        # Find a collision and fix it
        diffs = {}
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                d = U[j] - U[i]
                if d in diffs:
                    # Move one of the elements
                    idx = rng.choice([i, j])
                    old = U[idx]
                    new = old + rng.choice([-3, -2, -1, 1, 2, 3])
                    if new > 0 and new < max_val + 10 and new not in U:
                        U[idx] = new
                        U.sort()
                else:
                    diffs[d] = (i, j)
    return U

def _simulated_annealing(U, deadline, rng, T0=5.0, cooling=0.995):
    """Simulated annealing to optimize the set."""
    U = sorted(U)
    current_score = _score(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    T = T0
    
    while time.time() < deadline and T > 0.01:
        # Choose a random move
        move_type = rng.random()
        n = len(U)
        
        if move_type < 0.5 and n > 1:
            # Move one element
            i = rng.randint(1, n - 1)
            delta = rng.randint(-5, 5)
            if delta == 0:
                continue
            new_val = U[i] + delta
            if new_val <= 0 or new_val in U:
                continue
            new_U = U[:]
            new_U[i] = new_val
            new_U.sort()
        elif move_type < 0.7 and n > 2:
            # Swap two elements
            i, j = rng.sample(range(1, n), 2)
            new_U = U[:]
            new_U[i], new_U[j] = new_U[j], new_U[i]
            new_U.sort()
        elif move_type < 0.85 and n < 12:
            # Add an element
            max_u = U[-1]
            new_val = rng.randint(1, max_u + 20)
            if new_val not in U:
                new_U = sorted(U + [new_val])
            else:
                continue
        elif move_type < 0.95 and n > 3:
            # Remove an element
            i = rng.randint(1, n - 1)
            new_U = [U[j] for j in range(n) if j != i]
        else:
            # Random restart with small perturbation
            new_U = U[:]
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(1, len(new_U) - 1)
                delta = rng.randint(-10, 10)
                if delta != 0:
                    new_val = new_U[i] + delta
                    if new_val > 0 and new_val not in new_U:
                        new_U[i] = new_val
                        new_U.sort()
        
        new_score = _score(new_U)
        if new_score < 0:
            new_score = -1
        
        # Accept if better or with probability exp((new - old)/T)
        if new_score >= current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        else:
            prob = exp_val = np.exp((new_score - current_score) / T) if T > 0.01 else 0
            if rng.random() < prob:
                U = new_U
                current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def _local_search(U, deadline, rng):
    """Greedy local search with multiple passes."""
    U = sorted(U)
    current_score = _score(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(U)
        
        # Try moving each element
        for i in range(1, n):
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_val = U[i] + delta
                if new_val <= 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                new_score = _score(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 12:
            max_u = U[-1]
            for new_val in range(1, max_u + 30):
                if new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        
        # Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = _score(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        # Try pairwise swaps
        if not improved:
            for i in range(1, n):
                for j in range(i + 1, n):
                    new_U = U[:]
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    new_U.sort()
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_U = U[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(12345)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = _score(best_set)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Known good sets
    known = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 9, 11, 13],
        [0, 1, 4, 9, 11, 13, 15],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 3, 7, 12, 18],
        [0, 1, 4, 7, 10, 13, 16],
        [0, 1, 5, 9, 13, 17, 21],
        [0, 2, 3, 7, 12, 18, 25],
        [0, 1, 4, 8, 12, 16, 20],
        [0, 1, 3, 4, 9, 11],
        [0, 1, 4, 10, 12, 17, 23],
    ]
    for k in known:
        candidates.append(k)
        
    # Add random sets to diversify
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        n = rng.randint(3, 15)
        max_val = rng.randint(20, 100)
        U = [0] + sorted(rng.sample(range(1, max_val + 1), min(n - 1, max_val)))
        candidates.append(U)

    # Evaluate all candidates
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 50:
            break
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
            
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Simulated annealing from best candidates
    top_candidates = [c for _, c in scored_candidates[:min(15, len(scored_candidates))]]
    if best_set not in top_candidates:
        top_candidates.append(best_set)
        
    # Run SA on top candidates
    for init in top_candidates:
        if time.time() > deadline - 20:
            break
        # Allocate remaining time split among candidates? 
        # Simpler: run SA with a short deadline for each, or one long SA on the best.
        # Let's run a few shorter SAs.
        sa_deadline = min(deadline - 5, time.time() + 5) # 5s per SA attempt
        result, score = _simulated_annealing(init, sa_deadline, rng, T0=5.0, cooling=0.995)
        if score > best_score:
            best_score = score
            best_set = result[:]
            
    # Phase 3: Local search polish on best
    if time.time() < deadline - 2:
        result, score = _local_search(best_set, deadline - 1, rng)
        if score > best_score:
            best_score = score
            best_set = result[:]
            
    # Final verification
    final_score = _score(best_set)
    if final_score < 0:
        best_set = [0, 1, 4, 9, 11]
        
    return best_set

# EVOLVE-BLOCK-END
