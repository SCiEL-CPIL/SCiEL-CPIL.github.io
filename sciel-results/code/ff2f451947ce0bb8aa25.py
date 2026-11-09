"""Rewritten program with a fundamentally different algorithmic approach.

Instead of generic simulated annealing with full re-evaluation, this uses:
1. Fast vectorized evaluation via residue-class prefix sums
2. Targeted local search with incremental discrepancy tracking
3. Multi-start with structured initializations (Thue-Morse, balanced, random)
4. Iterated local search with restarts when stuck
"""
import time
import random
import numpy as np


def _build_progression_membership(n=256):
    """Build a data structure for fast discrepancy evaluation.
    
    For each step d, we organize positions by residue class mod d.
    This allows O(1) sum queries for any arithmetic progression.
    """
    # For each step d (1..n), for each residue r (0..d-1), store the list of positions
    # with that residue mod d, and a prefix sum array for fast range sums.
    residue_data = {}
    for d in range(1, n + 1):
        residue_data[d] = {}
        for r in range(d):
            positions = list(range(r, n, d))
            if not positions:
                residue_data[d][r] = (positions, None, None)
            else:
                residue_data[d][r] = (positions, None, None)
    return residue_data


def _compute_all_progression_sums(signs, n=256):
    """Compute the sum for every arithmetic progression.
    
    Returns a list of (step, start, length, sum) tuples, or just the max abs sum.
    """
    max_abs = 0
    for d in range(1, n + 1):
        for start in range(0, n, d):
            # Progression: start, start+d, start+2d, ...
            indices = range(start, n, d)
            s = 0
            for idx in indices:
                s += signs[idx]
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = abs_s
    return max_abs


def _fast_max_discrepancy(signs, n=256):
    """Compute max |sum| over all APs using prefix sums per residue class.
    
    For each step d and residue r, we have positions p_0 < p_1 < ... < p_{k-1}.
    The sum of any contiguous subsequence is a range sum, which we can compute
    via prefix sums in O(1). But we need ALL contiguous subsequences, which is
    O(k^2) per residue class. Total is O(n^2 log n) roughly.
    
    Actually, let's just iterate over all APs directly but use numpy for speed.
    """
    signs = np.array(signs, dtype=np.int8)
    max_disc = 0
    for d in range(1, n + 1):
        for start in range(0, n, d):
            idx = np.arange(start, n, d)
            s = int(np.sum(signs[idx]))
            if abs(s) > max_disc:
                max_disc = abs(s)
    return max_disc


def _max_disc_with_progression_info(signs, n=256):
    """Return (max_disc, list_of_worst_progressions) for targeted search."""
    signs = np.array(signs, dtype=np.int8)
    max_disc = 0
    worst_list = []
    for d in range(1, n + 1):
        for start in range(0, n, d):
            idx = np.arange(start, n, d)
            s = int(np.sum(signs[idx]))
            abs_s = abs(s)
            if abs_s > max_disc:
                max_disc = abs_s
                worst_list = [(d, start, len(idx), s)]
            elif abs_s == max_disc and max_disc > 0:
                worst_list.append((d, start, len(idx), s))
    return max_disc, worst_list


def _thue_morse_init(n=256):
    """Thue-Morse sequence: sign = +1 if popcount(i) is even, -1 otherwise."""
    signs = []
    for i in range(n):
        bc = bin(i).count('1')
        signs.append(1 if bc % 2 == 0 else -1)
    return signs


def _balanced_random_init(n=256, rng=None):
    """Random signs with exactly n/2 of each."""
    if rng is None:
        rng = random.Random()
    signs = [1] * (n // 2) + [-1] * (n // 2)
    rng.shuffle(signs)
    return signs


def _greedy_constructive(n=256, rng=None):
    """Greedy: pick each sign to minimize the max discrepancy so far."""
    if rng is None:
        rng = random.Random()
    signs = [0] * n
    
    for pos in range(n):
        best_sign = None
        best_max = 0
        for candidate in [1, -1]:
            signs[pos] = candidate
            # Check all APs containing pos
            current_max = 0
            for d in range(1, n + 1):
                for start in range(0, pos + 1, d):
                    if (pos - start) % d == 0:
                        idx = [i for i in range(start, pos + 1, d)]
                        # Only count positions that are assigned
                        assigned = [signs[i] for i in idx if signs[i] != 0]
                        if len(assigned) == len(idx):
                            s = sum(assigned)
                            if abs(s) > current_max:
                                current_max = abs(s)
            if best_sign is None or current_max < best_max:
                best_max = current_max
                best_sign = candidate
            elif current_max == best_max and rng.random() < 0.5:
                best_sign = candidate
        signs[pos] = best_sign
    
    return signs


def _local_search(signs, n=256, max_iters=20000, deadline=None, rng=None):
    """Targeted local search: flip positions in currently-worst APs."""
    if rng is None:
        rng = random.Random()
    signs = np.array(signs, dtype=np.int8)
    current_disc = _fast_max_discrepancy(signs.tolist(), n)
    best_disc = current_disc
    best_signs = signs.copy()
    
    iteration = 0
    while iteration < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iteration += 1
        
        # Find current worst progressions
        disc, worst = _max_disc_with_progression_info(signs.tolist(), n)
        current_disc = disc
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
        
        if best_disc <= 3:
            break
        
        # Pick a random position from a random worst progression
        if worst:
            d, start, length, s = rng.choice(worst)
            # Pick a random position in this progression
            idx_in_ap = rng.randint(0, length - 1)
            pos = start + idx_in_ap * d
        else:
            pos = rng.randint(0, n - 1)
        
        # Try flipping this position
        old = signs[pos]
        signs[pos] = -old
        new_disc = _fast_max_discrepancy(signs.tolist(), n)
        
        if new_disc <= current_disc:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            # Accept with small probability (simulated annealing flavor)
            if rng.random() < 0.05:
                current_disc = new_disc
            else:
                signs[pos] = old
    
    return best_signs.tolist(), best_disc


def _iterated_local_search(n=256, deadline=None, rng=None):
    """Multi-start local search with perturbations."""
    if rng is None:
        rng = random.Random()
    
    best_overall = None
    best_disc = float('inf')
    
    # Initializations
    initializations = []
    initializations.append(('thue_morse', _thue_morse_init(n)))
    initializations.append(('balanced', _balanced_random_init(n, rng)))
    initializations.append(('greedy', _greedy_constructive(n, rng)))
    # A few random ones
    for _ in range(2):
        initializations.append(('random', _balanced_random_init(n, rng)))
    
    for name, init_signs in initializations:
        if deadline is not None and time.time() > deadline:
            break
        sub_deadline = deadline
        if deadline is not None:
            remaining = deadline - time.time()
            sub_deadline = time.time() + max(1.0, remaining * 0.4)
        
        signs, disc = _local_search(init_signs, n, max_iters=5000, deadline=sub_deadline, rng=rng)
        
        if disc < best_disc:
            best_disc = disc
            best_overall = signs
        
        # If we have time, do a perturbation and re-search
        if deadline is not None and time.time() < deadline - 5:
            # Perturb: flip a few random positions
            perturbed = signs.copy()
            for _ in range(rng.randint(3, 10)):
                p = rng.randint(0, n - 1)
                perturbed[p] = -perturbed[p]
            
            pert_deadline = deadline
            if deadline is not None:
                remaining = deadline - time.time()
                pert_deadline = time.time() + max(1.0, remaining * 0.3)
            
            signs2, disc2 = _local_search(perturbed, n, max_iters=3000, deadline=pert_deadline, rng=rng)
            if disc2 < best_disc:
                best_disc = disc2
                best_overall = signs2
    
    if best_overall is None:
        best_overall = _thue_morse_init(n)
        best_disc = _fast_max_discrepancy(best_overall, n)
    
    return best_overall, best_disc


def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 90s limit
    
    rng = random.Random(12345)
    n = 256
    
    best_signs, best_disc = _iterated_local_search(n, deadline, rng)
    
    # Final check
    if time.time() < deadline:
        # One more pass with the best found
        final_deadline = start_time + 55
        final_signs, final_disc = _local_search(best_signs, n, max_iters=10000, deadline=final_deadline, rng=rng)
        if final_disc < best_disc:
            best_signs = final_signs
            best_disc = final_disc
    
    return best_signs


# EVOLVE-BLOCK-END
