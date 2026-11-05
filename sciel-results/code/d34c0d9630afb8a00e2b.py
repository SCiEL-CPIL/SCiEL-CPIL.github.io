"""Pairwise joint optimization with multi-start for the covering problem."""
import time
import numpy as np
import random


def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts


def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]


def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)


def _evaluate_pair(residues, primes, i, j, new_r_i, new_r_j, counts, positions, max_L):
    """Evaluate the score if we change residue i to new_r_i and j to new_r_j."""
    # Remove old contributions
    old_pos_i = positions[i][residues[i]]
    old_pos_j = positions[j][residues[j]]
    
    # Create a temporary copy to avoid modifying the main counts
    # But copying the whole array is expensive. Let's do it in place and revert.
    # Actually, for a pairwise evaluation, we can just compute the delta.
    # But the interaction makes it complex.
    # Let's just do a full recomputation for the pair evaluation if it's not too slow,
    # or use a smarter delta.
    
    # Smart delta:
    # counts_new = counts - old_i - old_j + new_i + new_j
    # We need to find the first x where counts_new[x] < 2.
    
    # To avoid copying the whole array, we can check the affected positions.
    # But the "first x" logic requires scanning from 1 upwards.
    # If the current L is small, the affected positions might be far.
    
    # Let's just copy. It's 100k ints, which is fast.
    c = counts.copy()
    
    if len(old_pos_i) > 0:
        c[old_pos_i] -= 1
    if len(old_pos_j) > 0:
        c[old_pos_j] -= 1
        
    new_pos_i = positions[i][new_r_i]
    new_pos_j = positions[j][new_r_j]
    
    if len(new_pos_i) > 0:
        c[new_pos_i] += 1
    if len(new_pos_j) > 0:
        c[new_pos_j] += 1
        
    return _find_L(c, max_L)


def _greedy_construct(primes, max_L):
    """Greedy forward construction."""
    residues = []
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for p in primes:
        best_r = 0
        best_L = -1
        for r in range(p):
            # Tentatively add r
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L:
                pos = np.arange(start, max_L + 1, p)
                if len(pos) > 0:
                    counts[pos] += 1
                    L = _find_L(counts, max_L)
                    if L > best_L:
                        best_L = L
                        best_r = r
                    counts[pos] -= 1
        residues.append(best_r)
        # Commit best_r
        if best_r == 0:
            start = p
        else:
            start = best_r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
    return residues


def _random_construct(primes):
    """Random initialization."""
    return [random.randint(0, p - 1) for p in primes]


def _pattern_construct(primes):
    """Pattern-based initialization (e.g., r = p//2)."""
    return [(p // 2) % p for p in primes]


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000

    # Precompute positions
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            pos_list.append(_get_positions(r, p, max_L))
        positions.append(pos_list)

    # Multi-start
    initializations = []
    
    # 1. Greedy
    initializations.append(_greedy_construct(primes, max_L))
    
    # 2. Random
    for _ in range(3):
        initializations.append(_random_construct(primes))
        
    # 3. Pattern
    initializations.append(_pattern_construct(primes))
    
    # 4. Zero
    initializations.append([0] * n_primes)
    
    # 5. Small offset
    initializations.append([(i + 1) % p for i, p in enumerate(primes)])

    best_residues = None
    best_L = -1

    for init in initializations:
        if time.time() > deadline - 30:  # Leave time for at least one optimization pass
            break
            
        residues = init[:]
        counts = _compute_counts(residues, primes, max_L)
        current_L = _find_L(counts, max_L)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]
            best_counts = counts.copy()
        else:
            best_counts = counts  # We will optimize this one too, but keep the global best separate

        # Optimize this specific start
        # Phase 1: Single-variable hill climbing
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            for i in range(n_primes):
                if time.time() > deadline - 10:
                    break
                p = primes[i]
                current_r = residues[i]
                old_pos = positions[i][current_r]
                
                # Base state: remove current
                base = counts.copy()
                if len(old_pos) > 0:
                    base[old_pos] -= 1
                    
                best_new_r = current_r
                best_new_L = current_L
                
                # Evaluate all candidates
                # For speed, we can sample if p is large, but let's try all for small p
                # p <= 1000, so max 1000 candidates.
                # 1000 * 100k ops = 100M ops. Might be slow in Python loop.
                # Let's vectorize the evaluation for a single prime change.
                
                # Create a (p, max_L+1) array? No, too big.
                # Let's just loop but break early if we find max_L.
                # Or use the batch method from the seed program?
                # The seed program used batch. Let's stick to a simpler vectorized check.
                
                # Actually, let's just do the loop but optimize the inner part.
                for new_r in range(p):
                    if new_r == current_r:
                        continue
                    new_pos = positions[i][new_r]
                    # c = base + new_pos
                    # We need to find L.
                    # If new_pos is empty, L is _find_L(base)
                    # If not, it's complex.
                    
                    # Let's just do the copy method for simplicity in this step,
                    # and rely on the pairwise step for the heavy lifting.
                    # Wait, 168 primes * 1000 candidates * 100k array copy = 16.8B ops. Too slow.
                    
                    # We MUST use the batch/vectorized method from the seed for the single variable step.
                    # Let's implement a fast single-variable optimization.
                    
                # Re-implementing the fast single variable step from seed program logic
                # but integrated here.
                
                # Batch size 50
                BATCH_SIZE = 50
                candidates = [r for r in range(p) if r != current_r]
                for batch_start in range(0, len(candidates), BATCH_SIZE):
                    if time.time() > deadline - 10:
                        break
                    batch = candidates[batch_start:batch_start + BATCH_SIZE]
                    B = len(batch)
                    
                    batch_arr = np.tile(base, (B, 1))
                    for j, new_r in enumerate(batch):
                        new_pos = positions[i][new_r]
                        if len(new_pos) > 0:
                            batch_arr[j, new_pos] += 1
                            
                    below_two_mask = batch_arr[:, 1:] < 2
                    first_below = np.argmax(below_two_mask, axis=1)
                    has_below = below_two_mask.any(axis=1)
                    Ls = np.where(has_below, first_below + 1, max_L)
                    
                    best_idx_in_batch = np.argmax(Ls)
                    best_L_in_batch = Ls[best_idx_in_batch]
                    if best_L_in_batch > best_new_L:
                        best_new_L = best_L_in_batch
                        best_new_r = batch[best_idx_in_batch]
                        
                if best_new_r != current_r:
                    # Apply
                    if len(old_pos) > 0:
                        counts[old_pos] -= 1
                    new_pos = positions[i][best_new_r]
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                    residues[i] = best_new_r
                    current_L = best_new_L
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                        best_counts = counts.copy()

        # Phase 2: Pairwise joint optimization
        # Focus on pairs (i, j) where i is small prime, j is large prime
        # Or just random pairs.
        # Let's do a few passes of random pair optimization.
        for _ in range(5):  # 5 passes
            if time.time() > deadline - 5:
                break
            # Shuffle prime indices to vary the order
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for k in range(0, n_primes - 1, 2):
                if time.time() > deadline - 5:
                    break
                i = indices[k]
                j = indices[k+1]
                
                p_i = primes[i]
                p_j = primes[j]
                
                r_i = residues[i]
                r_j = residues[j]
                
                # We want to find (new_r_i, new_r_j) that maximizes L.
                # Brute force p_i * p_j is too big (1000*1000 = 1M).
                # Let's fix one and optimize the other, then swap.
                
                # Optimize j given i
                best_j = r_j
                best_L_j = current_L
                # Sample candidates for j? Or all?
                # Let's sample 50 candidates for j
                candidates_j = random.sample(range(p_j), min(50, p_j))
                if r_j not in candidates_j:
                    candidates_j.append(r_j)
                    
                for new_r_j in candidates_j:
                    L = _evaluate_pair(residues, primes, i, j, r_i, new_r_j, counts, positions, max_L)
                    if L > best_L_j:
                        best_L_j = L
                        best_j = new_r_j
                        
                if best_j != r_j:
                    # Update counts and residues for j
                    old_pos_j = positions[j][r_j]
                    new_pos_j = positions[j][best_j]
                    if len(old_pos_j) > 0:
                        counts[old_pos_j] -= 1
                    if len(new_pos_j) > 0:
                        counts[new_pos_j] += 1
                    residues[j] = best_j
                    current_L = best_L_j
                    
                # Optimize i given new j
                best_i = r_i
                best_L_i = current_L
                candidates_i = random.sample(range(p_i), min(50, p_i))
                if r_i not in candidates_i:
                    candidates_i.append(r_i)
                
                for new_r_i in candidates_i:
                    L = _evaluate_pair(residues, primes, i, j, new_r_i, residues[j], counts, positions, max_L)
                    if L > best_L_i:
                        best_L_i = L
                        best_i = new_r_i
                        
                if best_i != r_i:
                    # Update counts and residues for i
                    old_pos_i = positions[i][r_i]
                    new_pos_i = positions[i][best_i]
                    if len(old_pos_i) > 0:
                        counts[old_pos_i] -= 1
                    if len(new_pos_i) > 0:
                        counts[new_pos_i] += 1
                    residues[i] = best_i
                    current_L = best_L_i

                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                    best_counts = counts.copy()

    return best_residues
