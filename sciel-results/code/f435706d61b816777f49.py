import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = int(residues[i])
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _apply_move(count, residues, i, new_r):
    p = _PRIMES[i]
    old_r = int(residues[i])
    if old_r == new_r:
        return
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    count[start_old::p] -= 1
    count[start_new::p] += 1
    residues[i] = new_r

def _revert_move(count, residues, i, old_r):
    p = _PRIMES[i]
    new_r = int(residues[i])
    start_new = p if new_r == 0 else new_r
    start_old = p if old_r == 0 else old_r
    count[start_new::p] -= 1
    count[start_old::p] += 1
    residues[i] = old_r

def _maxmin_greedy_init(L_TARGET=2000):
    """
    Greedy initialization maximizing the minimum coverage count.
    Process primes in increasing order. For each prime, pick the residue
    that maximizes the minimum count in [1, L_TARGET] after adding it.
    """
    residues = [0] * _NPRIMES
    count = np.zeros(L_TARGET + 1, dtype=np.int32)
    
    for i, p in enumerate(_PRIMES):
        if p > L_TARGET:
            # For large primes, just pick a random residue; they contribute little
            residues[i] = random.randint(0, p - 1)
            continue
        
        best_r = 0
        best_min = -1
        best_sum = -1
        
        # Sample a subset of residues for speed
        if p <= 50:
            candidates = list(range(p))
        else:
            candidates = list(range(0, p, max(1, p // 100)))
            if 0 not in candidates:
                candidates.append(0)
        
        for r in candidates:
            start = p if r == 0 else r
            if start > L_TARGET:
                continue
            
            positions = np.arange(start, L_TARGET + 1, p)
            if len(positions) == 0:
                continue
            
            new_counts = count[positions] + 1
            # We want to maximize the minimum of (count + indicator)
            # Approximate: look at positions where count is currently 0 or 1
            min_after = np.min(new_counts)
            sum_after = np.sum(new_counts)
            
            # Prefer residues that improve the weakest positions
            # Use a composite score: weight min heavily
            score = min_after * 1000 + sum_after
            
            if score > best_min or (score == best_min and sum_after > best_sum):
                best_min = score
                best_sum = sum_after
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_TARGET:
            count[start::p] += 1
    
    return residues

def _conflict_repair(count, residues, L_TARGET, max_iter=5000, time_limit=10.0):
    """
    Conflict-driven repair: repeatedly find the first gap, then choose
    the prime+residue change that covers it with minimal damage.
    """
    start_time = time.time()
    current_L = _find_prefix_L_np(count, L_TARGET)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        gap = current_L + 1
        if gap > L_TARGET:
            break
        
        if count[gap] >= 2:
            # Advance past covered region
            while gap <= L_TARGET and count[gap] >= 2:
                current_L = gap
                gap += 1
            continue
        
        # Find best move to cover gap
        best_move = None
        best_score = float('-inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            old_r = int(residues[i])
            new_r = gap % p
            
            if old_r == new_r:
                continue
            
            # Check if this move actually covers the gap
            start_new = p if new_r == 0 else new_r
            if start_new > L_TARGET:
                continue
            
            # The gap is covered if gap >= start_new and (gap - start_new) % p == 0
            # Since new_r = gap % p, this is true by construction if start_new <= gap
            
            if start_new > gap:
                continue
            
            # Calculate benefit and cost
            # Benefit: number of positions in [1, current_L] that go from 1 to 2
            # Cost: number of positions in [1, current_L] that go from 2 to 1
            
            start_old = p if old_r == 0 else old_r
            
            # Positions affected by removing old_r
            old_positions = np.arange(start_old, L_TARGET + 1, p) if start_old <= L_TARGET else np.array([], dtype=np.int64)
            # Positions affected by adding new_r
            new_positions = np.arange(start_new, L_TARGET + 1, p) if start_new <= L_TARGET else np.array([], dtype=np.int64)
            
            # Net effect on count
            # For positions in both old and new: no change
            # For positions only in old: count decreases by 1
            # For positions only in new: count increases by 1
            
            # Simplified: just look at the gap and nearby
            # Cost: positions in [1, current_L] that are in old_positions but not new_positions
            # and currently have count == 2
            # Benefit: positions in [1, current_L] that are in new_positions but not old_positions
            # and currently have count == 1
            
            # For efficiency, just compute directly
            if len(old_positions) > 0 and len(new_positions) > 0:
                old_set = set(old_positions.tolist())
                new_set = set(new_positions.tolist())
                only_old = old_set - new_set
                only_new = new_set - old_set
            elif len(old_positions) > 0:
                only_old = set(old_positions.tolist())
                only_new = set()
            elif len(new_positions) > 0:
                only_old = set()
                only_new = set(new_positions.tolist())
            else:
                continue
            
            cost = 0
            benefit = 0
            
            if only_old:
                old_arr = np.array(list(only_old), dtype=np.int64)
                old_arr = old_arr[old_arr <= current_L]
                if len(old_arr) > 0:
                    cost = int(np.sum(count[old_arr] == 2))
            
            if only_new:
                new_arr = np.array(list(only_new), dtype=np.int64)
                new_arr = new_arr[new_arr <= current_L]
                if len(new_arr) > 0:
                    benefit = int(np.sum(count[new_arr] == 1))
            
            # Also check if the gap itself gets covered
            gap_covered = 1 if count[gap] + (1 if gap in new_set else 0) - (1 if gap in old_set else 0) >= 2 else 0
            
            # Score: strongly prefer covering the gap, then maximize net benefit
            score = gap_covered * 10000 + benefit - cost * 2
            
            if score > best_score:
                best_score = score
                best_move = (i, new_r)
        
        if best_move is None:
            # Try a random perturbation
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            old_r = int(residues[i])
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            _apply_move(count, residues, i, new_r)
            new_L = _find_prefix_L_np(count, L_TARGET)
            if new_L < current_L - 5:
                _revert_move(count, residues, i, old_r)
            else:
                current_L = new_L
            continue
        
        i, new_r = best_move
        old_r = int(residues[i])
        _apply_move(count, residues, i, new_r)
        
        new_L = _find_prefix_L_np(count, L_TARGET)
        if new_L > current_L:
            current_L = new_L
        else:
            _revert_move(count, residues, i, old_r)
            # Try a small perturbation to escape
            for _ in range(3):
                j = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[j]
                old_r2 = int(residues[j])
                new_r2 = random.randint(0, p - 1)
                if new_r2 == old_r2:
                    continue
                _apply_move(count, residues, j, new_r2)
                new_L2 = _find_prefix_L_np(count, L_TARGET)
                if new_L2 >= current_L:
                    current_L = new_L2
                    break
                else:
                    _revert_move(count, residues, j, old_r2)
    
    return current_L

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    random.seed(42)
    
    # Phase 1: Greedy initialization with max-min objective
    L_INIT = 2000
    residues = _maxmin_greedy_init(L_INIT)
    
    # Phase 2: Build count and find initial L
    L_LIMIT = 5000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    
    best_L = current_L
    best_residues = [int(r) for r in residues]
    
    # Phase 3: Conflict-driven repair with increasing L_TARGET
    L_TARGET = L_LIMIT
    while time.time() < deadline - 10:
        # Run conflict repair
        current_L = _conflict_repair(count, residues, L_TARGET, max_iter=3000, time_limit=8.0)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = [int(r) for r in residues]
        
        # Check if we've reached the limit
        if current_L >= L_TARGET - 10:
            # Extend the window
            new_limit = min(L_TARGET * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_TARGET = new_limit
            current_L = _find_prefix_L_np(count, L_TARGET)
        else:
            # Stagnated: try a different strategy
            # Perturb some residues and retry
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = int(residues[i])
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                _apply_move(count, residues, i, new_r)
                new_L = _find_prefix_L_np(count, L_TARGET)
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = [int(r) for r in residues]
                else:
                    _revert_move(count, residues, i, old_r)
            
            # If still stuck, try targeted moves on the gap
            gap = current_L + 1
            if gap <= L_TARGET:
                for i in range(_NPRIMES):
                    p = _PRIMES[i]
                    old_r = int(residues[i])
                    new_r = gap % p
                    if old_r == new_r:
                        continue
                    _apply_move(count, residues, i, new_r)
                    new_L = _find_prefix_L_np(count, L_TARGET)
                    if new_L > current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = [int(r) for r in residues]
                        break
                    else:
                        _revert_move(count, residues, i, old_r)
    
    # Final: try to extend beyond current best
    if time.time() < deadline - 5:
        L_FINAL = min(100000, best_L * 2 + 1000)
        count_final = _build_count_np(best_residues, L_FINAL)
        L_check = _find_prefix_L_np(count_final, L_FINAL)
        if L_check > best_L:
            best_L = L_check
            best_residues = [int(r) for r in residues]
    
    # Final verification
    if time.time() < deadline - 2:
        count_verify = _build_count_np(best_residues, 100000)
        final_L = _find_prefix_L_np(count_verify, 100000)
        # Just return best_residues; the evaluator will compute the score
    
    return [int(r) for r in best_residues]
