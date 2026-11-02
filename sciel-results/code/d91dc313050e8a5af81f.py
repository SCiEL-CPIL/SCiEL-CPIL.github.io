"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Strategy: Use a modular lattice representation where elements are selected to
form a dense sublattice in Z_m for a carefully chosen modulus m. The key insight
is that if U is contained in a small subgroup or coset structure modulo m, then
U+U can collapse while U-U remains large. We also combine this with Golomb-ruler
based constructions and local search refinement.
"""
import time
import math
import random
import numpy as np


def _score_fast(U):
    """Fast score computation using numpy."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _score_fast_exact(U):
    """Exact score computation for small sets."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(n):
            sums.add(U[i] + U[j])
            diffs.add(U[i] - U[j])
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _modular_lattice_set(m, density, max_elements=50):
    """
    Construct a set U using modular lattice structure.
    
    We pick elements that are multiples of a step 's' modulo m, creating
    a structured set where sums collapse modulo m but differences spread out.
    
    Strategy: U = {0, s, 2s, ..., (k-1)s} where s is chosen so that
    the sumset has fewer distinct values than the difference set.
    """
    # Try various step sizes
    best_U = None
    best_score = -1.0
    
    for s in range(1, m + 1):
        if s >= m:
            break
        # Elements: 0, s, 2s, ..., up to max_elements
        U = []
        for i in range(max_elements):
            val = i * s
            if val > 2 * m * 10:  # Don't go too far
                break
            U.append(val)
        
        if len(U) < 2:
            continue
        
        # Also try: elements that are multiples of s but with gaps
        # to maximize difference spread
        score = _score_fast(U)
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    return best_U, best_score


def _arithmetic_progression_set(start, step, length, offset=0):
    """Construct an arithmetic progression set."""
    U = [offset]
    for i in range(1, length):
        U.append(offset + i * step)
    return U


def _mixed_structure_set(params):
    """
    Construct a set with mixed structure:
    - A base arithmetic progression
    - Plus some sparse large elements to increase difference spread
    """
    base_len = params.get('base_len', 10)
    base_step = params.get('base_step', 1)
    sparse_count = params.get('sparse_count', 3)
    sparse_scale = params.get('sparse_scale', 100)
    
    U = [0]
    # Base arithmetic progression
    for i in range(1, base_len):
        U.append(i * base_step)
    
    # Add sparse large elements
    max_base = (base_len - 1) * base_step
    for i in range(sparse_count):
        U.append(max_base + (i + 1) * sparse_scale)
    
    return sorted(set(U))


def _geometric_set(base, length):
    """Construct a geometric set {0, 1, b, b^2, ..., b^(n-1)}."""
    U = [0]
    val = 1
    for i in range(1, length):
        val *= base
        U.append(val)
    return U


def _sparse_large_set(n_elements, max_val):
    """
    Construct a sparse set with large spread to maximize |U-U|.
    Uses a power-of-prime or similar construction.
    """
    U = [0]
    current = 0
    for i in range(1, n_elements):
        # Exponential growth
        current = int(current * 1.7 + random.randint(1, 5))
        if current > max_val:
            break
        U.append(current)
    return sorted(U)


def _local_search(U, time_limit=30.0, max_steps=80000):
    """Local search with add, remove, and swap operations."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start = time.time()
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        
        improved = False
        max_u = max(U) if U else 0
        
        # Try adding a new element
        candidates = []
        # Near max
        for d in range(1, 40):
            candidates.append(max_u + d)
        # Geometric extensions
        for d in range(1, 20):
            candidates.append(max_u * 2 + d)
            candidates.append(max_u * 3 + d)
            candidates.append(max_u * 4 + d)
        # Interior random
        for _ in range(15):
            candidates.append(random.randint(1, max_u))
        # Large jumps
        for _ in range(5):
            candidates.append(random.randint(max_u, max_u * 3 + 100))
        
        random.shuffle(candidates)
        for new_val in candidates[:15]:
            if new_val <= 0 or new_val in U:
                continue
            new_U = sorted(U + [new_val])
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                improved = True
                break
        
        if not improved:
            # Try removing an element (not 0)
            if len(U) > 2:
                idxs = list(range(1, len(U)))
                random.shuffle(idxs)
                for idx in idxs[:8]:
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True
                        break
        
        if not improved:
            # Try swapping an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Try several candidate replacements
                found = False
                for _ in range(5):
                    new_val = random.randint(1, max_u * 2 + 100)
                    if new_val != old_val and new_val not in U:
                        new_U = list(U)
                        new_U[idx] = new_val
                        new_U = sorted(new_U)
                        new_score = _score_fast(new_U)
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_U = new_U
                            U = new_U
                            improved = True
                            found = True
                            break
        
        if not improved and step > 300 and random.random() < 0.03:
            # Perturb: replace a random element with a random value
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_val = random.randint(1, max_u * 3)
                if new_val not in U:
                    U = list(U)
                    U[idx] = new_val
                    U = sorted(U)
                    best_score = _score_fast(U)
                    best_U = list(U)
    
    return best_U, best_score


def _try_modular_construction(deadline):
    """Try modular lattice constructions."""
    best_U = None
    best_score = -1.0
    
    # Try various moduli
    for m in range(3, 60):
        if time.time() > deadline:
            break
        
        # For each modulus, try different step sizes and densities
        for s in range(1, min(m, 20)):
            if time.time() > deadline:
                break
            
            # Construct: multiples of s, but only those < some bound
            U = []
            max_val = m * 5  # Bound
            for i in range(0, max_val // s + 1):
                U.append(i * s)
                if len(U) > 40:
                    break
            
            if len(U) < 2:
                continue
            
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
        
        # Also try: U = {x : x mod m in S} for small S
        for size in range(2, 6):
            if time.time() > deadline:
                break
            for _ in range(3):
                S = set(random.sample(range(m), min(size, m)))
                U = [0]
                for x in range(1, m * 4):
                    if x % m in S:
                        U.append(x)
                        if len(U) > 30:
                            break
                if len(U) < 2:
                    continue
                score = _score_fast(U)
                if score > best_score:
                    best_score = score
                    best_U = list(U)
    
    return best_U, best_score


def _try_structured_families(deadline):
    """Try various structured families of sets."""
    best_U = None
    best_score = -1.0
    
    # Geometric sets
    for b in range(2, 50):
        if time.time() > deadline:
            break
        for n in range(2, 15):
            if time.time() > deadline:
                break
            U = _geometric_set(b, n)
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
    
    # Arithmetic progressions with sparse additions
    for base_len in range(3, 25):
        if time.time() > deadline:
            break
        for base_step in range(1, 20):
            if time.time() > deadline:
                break
            for sparse_count in range(0, 5):
                if time.time() > deadline:
                    break
                for sparse_scale in range(50, 500, 50):
                    if time.time() > deadline:
                        break
                    params = {
                        'base_len': base_len,
                        'base_step': base_step,
                        'sparse_count': sparse_count,
                        'sparse_scale': sparse_scale
                    }
                    U = _mixed_structure_set(params)
                    score = _score_fast(U)
                    if score > best_score:
                        best_score = score
                        best_U = list(U)
    
    # Sparse large sets
    for n_el in range(3, 15):
        if time.time() > deadline:
            break
        for max_val in [100, 500, 1000, 5000]:
            if time.time() > deadline:
                break
            for _ in range(5):
                U = _sparse_large_set(n_el, max_val)
                if len(U) < 2:
                    continue
                score = _score_fast(U)
                if score > best_score:
                    best_score = score
                    best_U = list(U)
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (evaluator kills at 150s)
    
    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)
    
    # Phase 1: Modular lattice constructions
    if time.time() < deadline - 60:
        U, score = _try_modular_construction(deadline - 60)
        if U and score > best_score:
            best_score = score
            best_U = U
    
    # Phase 2: Structured families
    if time.time() < deadline - 40:
        U, score = _try_structured_families(deadline - 40)
        if U and score > best_score:
            best_score = score
            best_U = U
    
    # Phase 3: Local search from best seed
    if time.time() < deadline - 25:
        remaining = deadline - time.time() - 25
        U, score = _local_search(best_U, time_limit=remaining, max_steps=80000)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 4: Additional local search rounds with perturbation
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb
            U = list(best_U)
            if len(U) > 2:
                # Remove a random non-zero element
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            
            max_u = max(U) if U else 0
            # Add a new element
            for _ in range(3):
                new_val = random.randint(1, max_u + max(20, max_u // 2))
                if new_val not in U:
                    U.append(new_val)
                    break
            U = sorted(U)
            
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
            U, score = _local_search(U, time_limit=min(remaining, 15.0), max_steps=30000)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)
