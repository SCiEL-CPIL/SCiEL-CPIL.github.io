"""Optimized search for finite set U maximizing sumset/difference-set exponent."""
import time
import random
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def _score(U):
    """Fast score computation using numpy vectorization."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    a = np.array(U, dtype=np.int64)
    # Compute all sums and diffs
    sums = a[:, None] + a[None, :]
    diffs = a[:, None] - a[None, :]
    
    # Count unique values efficiently
    n_sums = len(np.unique(sums))
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

def _exhaustive_small_search(max_n=6, max_val=30):
    """Exhaustively search for best small sets."""
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Generate all combinations
    # This is limited to small sizes to keep time low
    # We use a recursive generation or itertools.combinations
    import itertools
    
    for n in range(2, max_n + 1):
        # We must include 0
        # Choose n-1 elements from 1..max_val
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    return best_set, best_score

def _simulated_annealing(U, deadline, rng, T0=8.0, cooling=0.993):
    """Simulated annealing to optimize the set."""
    U = sorted(U)
    current_score = _score(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    T = T0
    stall_count = 0
    
    while time.time() < deadline and T > 0.01:
        # Choose a random move
        move_type = rng.random()
        n = len(U)
        
        if move_type < 0.4 and n > 1:
            # Move one element (larger range)
            i = rng.randint(1, n - 1)
            delta = rng.randint(-15, 15)
            if delta == 0:
                continue
            new_val = U[i] + delta
            if new_val <= 0 or new_val in U:
                continue
            new_U = U[:]
            new_U[i] = new_val
            new_U.sort()
        elif move_type < 0.55 and n > 2:
            # Swap two elements
            i, j = rng.sample(range(1, n), 2)
            new_U = U[:]
            new_U[i], new_U[j] = new_U[j], new_U[i]
            new_U.sort()
        elif move_type < 0.7 and n < 15:
            # Add an element
            max_u = U[-1]
            new_val = rng.randint(1, max_u + 40)
            if new_val not in U:
                new_U = sorted(U + [new_val])
            else:
                continue
        elif move_type < 0.8 and n > 3:
            # Remove an element
            i = rng.randint(1, n - 1)
            new_U = [U[j] for j in range(n) if j != i]
        elif move_type < 0.9:
            # Move two elements simultaneously
            if n < 2:
                continue
            i, j = rng.sample(range(1, n), 2)
            di = rng.randint(-10, 10)
            dj = rng.randint(-10, 10)
            if di == 0 and dj == 0:
                continue
            new_U = U[:]
            if di != 0:
                nv = new_U[i] + di
                if nv <= 0 or nv in new_U:
                    continue
                new_U[i] = nv
            if dj != 0:
                nv = new_U[j] + dj
                if nv <= 0 or nv in new_U:
                    continue
                new_U[j] = nv
            new_U.sort()
        else:
            # Large jump restart
            new_U = U[:]
            for _ in range(rng.randint(2, 5)):
                i = rng.randint(1, len(new_U) - 1)
                delta = rng.randint(-25, 25)
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
                stall_count = 0
            else:
                stall_count += 1
        else:
            prob = np.exp((new_score - current_score) / T) if T > 0.01 else 0
            if rng.random() < prob:
                U = new_U
                current_score = new_score
                stall_count += 1
            else:
                stall_count += 1
        
        # Increase temperature if stalled too long
        if stall_count > 50:
            T = max(T, 3.0)
            stall_count = 0
        
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
    no_improve_rounds = 0
    while improved and time.time() < deadline:
        improved = False
        n = len(U)
        
        # Try moving each element with wider range
        for i in range(1, n):
            for delta in range(-25, 26):
                if delta == 0:
                    continue
                new_val = U[i] + delta
                if new_val <= 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                new_score = _score(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 18:
            max_u = U[-1]
            for new_val in range(1, max_u + 50):
                if new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = _score(new_U)
                    if new_score > current_score + 1e-12:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        
        # Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = _score(new_U)
                if new_score > current_score + 1e-12:
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
                    if new_score > current_score + 1e-12:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score + 1e-12:
            best_score = current_score
            best_U = U[:]
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds > 3:
                break
    
    return best_U, best_score

def _final_polish(U, deadline, rng):
    """High-precision polish with small step sizes."""
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
        
        # Fine-grained single element moves
        for i in range(1, n):
            for delta in range(-8, 9):
                if delta == 0:
                    continue
                new_val = U[i] + delta
                if new_val <= 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                new_score = _score(new_U)
                if new_score > current_score + 1e-10:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding small elements near existing ones
        if not improved and n < 20:
            for i in range(n):
                for offset in range(1, 6):
                    new_val = U[i] + offset
                    if new_val not in U and new_val > 0:
                        new_U = sorted(U + [new_val])
                        new_score = _score(new_U)
                        if new_score > current_score + 1e-10:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                    new_val = U[i] - offset
                    if new_val > 0 and new_val not in U:
                        new_U = sorted(U + [new_val])
                        new_score = _score(new_U)
                        if new_score > current_score + 1e-10:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        # Try removing elements
        if not improved and n > 3:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = _score(new_U)
                if new_score > current_score + 1e-10:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_U = U[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(12345)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = _score(best_set)
    
    # Phase 0: Exhaustive search for small sets to find strong cores
    small_set, small_score = _exhaustive_small_search(max_n=6, max_val=35)
    if small_score > best_score:
        best_score = small_score
        best_set = small_set[:]
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Known good sets
    known = [
        [0, 1, 3],
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
        [0, 1, 4, 9, 11, 13, 15, 18],
        [0, 1, 4, 10, 12, 17, 23, 31],
        [0, 1, 3, 4, 9, 11, 13, 16],
    ]
    for k in known:
        candidates.append(k)
        
    # Add random sets to diversify
    for _ in range(80):
        if time.time() > deadline - 100:
            break
        n = rng.randint(4, 18)
        max_val = rng.randint(30, 150)
        U = [0] + sorted(rng.sample(range(1, max_val + 1), min(n - 1, max_val)))
        candidates.append(U)

    # Evaluate all candidates
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 90:
            break
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
            
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Simulated annealing from best candidates
    top_candidates = [c for _, c in scored_candidates[:min(20, len(scored_candidates))]]
    if best_set not in top_candidates:
        top_candidates.append(best_set)
        
    # Run SA on top candidates with adaptive time allocation
    remaining_time = deadline - time.time()
    num_sa = min(len(top_candidates), max(3, int(remaining_time / 8)))
    for i, init in enumerate(top_candidates[:num_sa]):
        if time.time() > deadline - 30:
            break
        # Allocate time: more for earlier (better) candidates
        time_per_sa = max(3, (deadline - 30 - time.time()) / max(1, num_sa - i))
        sa_deadline = min(deadline - 30, time.time() + time_per_sa)
        result, score = _simulated_annealing(init, sa_deadline, rng, T0=8.0, cooling=0.993)
        if score > best_score:
            best_score = score
            best_set = result[:]
            
    # Phase 3: Local search polish on best
    if time.time() < deadline - 15:
        result, score = _local_search(best_set, deadline - 14, rng)
        if score > best_score:
            best_score = score
            best_set = result[:]
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2:
        result, score = _final_polish(best_set, deadline - 1, rng)
        if score > best_score:
            best_score = score
            best_set = result[:]
            
    # Final verification
    final_score = _score(best_set)
    if final_score < 0:
        best_set = [0, 1, 3]
        
    return best_set

# EVOLVE-BLOCK-END
