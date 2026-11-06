"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_L(counts, max_check=100000):
    """
    Find the largest L such that counts[1..L] >= 2.
    Returns max_check if all are >= 2.
    """
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _update_counts(counts, r, p, sign, max_check):
    """
    Update counts array by adding or subtracting 1 at positions r, r+p, ...
    sign: 1 for add, -1 for remove.
    """
    start = r if r > 0 else p
    if start <= max_check:
        counts[start::p] += sign

def _greedy_construct(primes, max_check=100000):
    """
    Greedy initialization: process primes in increasing order,
    picking residue to cover the current bottleneck.
    """
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    current_L = 0
    
    for i, p in enumerate(primes):
        bottleneck = current_L + 1
        if bottleneck > max_check:
            r = 0
        else:
            r = bottleneck % p
        residues[i] = r
        _update_counts(counts, r, p, 1, max_check)
        new_L = _get_L(counts, max_check)
        if new_L > current_L:
            current_L = new_L
    
    return residues, counts, current_L

def _dpll_backtrack(residues, primes, counts, max_check, deadline, 
                    target_L, max_nodes=50000):
    """
    DPLL-style backtracking search to extend coverage to target_L.
    
    Uses unit propagation: find positions with count < 2 in [1, target_L],
    and try to fix them by adjusting residues of primes that can cover them.
    
    Returns (success, new_residues, new_counts)
    """
    node_count = [0]
    
    def _find_weak_positions(counts_arr, lo, hi):
        """Find positions in [lo, hi] with count < 2."""
        if lo > hi:
            return []
        seg = counts_arr[lo:hi+1]
        weak = np.where(seg < 2)[0]
        return [int(w + lo) for w in weak]
    
    def _can_cover(pos, p, r):
        """Check if residue r for prime p covers position pos."""
        return pos % p == r
    
    def _search(res_arr, cnt_arr, depth):
        """Recursive backtracking search."""
        if time.time() > deadline:
            return False, None, None
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return False, None, None
        
        # Find the first weak position in [1, target_L]
        weak_pos = _find_weak_positions(cnt_arr, 1, target_L)
        
        if not weak_pos:
            return True, res_arr.copy(), cnt_arr.copy()
        
        pos = weak_pos[0]
        current_count = cnt_arr[pos]
        
        # Find primes that could cover this position
        # A prime p covers pos if pos % p == r_p
        # We need to find which primes currently DON'T cover pos but COULD
        # if we changed their residue
        
        # First, check which primes already cover pos
        covering_primes = []
        for i, p in enumerate(primes):
            if pos % p == res_arr[i]:
                covering_primes.append(i)
        
        # We need at least (2 - current_count) more covers
        needed = 2 - current_count
        
        if needed <= 0:
            # Already covered enough, move to next weak position
            # This shouldn't happen if we found weak positions correctly
            return _search(res_arr, cnt_arr, depth + 1)
        
        # Find primes that can be changed to cover pos
        # For prime i with residue r_i, it covers pos if pos % p_i == r_i
        # So we need to set r_i = pos % p_i
        candidates = []
        for i, p in enumerate(primes):
            if i in covering_primes:
                continue  # Already covers
            target_r = pos % p
            if target_r != res_arr[i]:
                candidates.append((i, target_r))
        
        if len(candidates) < needed:
            return False, None, None
        
        # Try changing up to 'needed' primes to cover pos
        # Use a simple heuristic: try primes in order of impact
        # Small primes affect more positions, so changing them has bigger impact
        
        # Sort candidates by prime size (smaller first) for better heuristic
        candidates.sort(key=lambda x: primes[x[0]])
        
        # Try the first candidate first (most likely to help)
        for i, new_r in candidates[:min(needed * 3, 10)]:
            old_r = res_arr[i]
            p = primes[i]
            
            # Apply change
            _update_counts(cnt_arr, old_r, p, -1, max_check)
            _update_counts(cnt_arr, new_r, p, 1, max_check)
            res_arr[i] = new_r
            
            # Recurse
            success, new_res, new_cnt = _search(res_arr, cnt_arr, depth + 1)
            if success:
                return True, new_res, new_cnt
            
            # Backtrack
            res_arr[i] = old_r
            _update_counts(cnt_arr, new_r, p, -1, max_check)
            _update_counts(cnt_arr, old_r, p, 1, max_check)
        
        return False, None, None
    
    # Start search
    success, new_res, new_cnt = _search(residues.copy(), counts.copy(), 0)
    
    if success:
        return True, new_res, new_cnt
    else:
        return False, residues.copy(), counts.copy()

def _hill_climb_with_backtrack(residues, primes, max_check=100000, deadline=None):
    """
    Combined hill-climbing and backtracking search.
    """
    n_primes = len(primes)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        _update_counts(counts, r, p, 1, max_check)
    
    current_L = _get_L(counts, max_check)
    best_residues = residues.copy()
    best_L = current_L
    
    # Phase 1: Standard hill climbing to get a good starting point
    for pass_num in range(30):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline and time.time() > deadline:
                return best_residues, best_L
                
            p = primes[i]
            current_r = residues[i]
            
            # Generate candidates
            bottleneck = current_L + 1
            if bottleneck <= max_check:
                target_r = bottleneck % p
            else:
                target_r = current_r
                
            candidates = {current_r, target_r}
            
            if p < 50:
                candidates.update(np.random.randint(0, p, size=10).tolist())
            else:
                candidates.update(np.random.randint(0, p, size=5).tolist())
                
            best_local_r = current_r
            best_local_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                    
                _update_counts(counts, current_r, p, -1, max_check)
                _update_counts(counts, r, p, 1, max_check)
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = r
                else:
                    _update_counts(counts, r, p, -1, max_check)
                    _update_counts(counts, current_r, p, 1, max_check)
                    
            if best_local_r != current_r:
                if best_local_L > current_L:
                    residues[i] = best_local_r
                    current_L = best_local_L
                    improved = True
                    best_residues = residues.copy()
                    best_L = current_L
                else:
                    _update_counts(counts, best_local_r, p, -1, max_check)
                    _update_counts(counts, current_r, p, 1, max_check)
                    
        if not improved:
            break
    
    # Phase 2: Backtracking to fix specific gaps
    # Try to extend L by fixing the first weak position
    attempts = 0
    while attempts < 20:
        if deadline and time.time() > deadline:
            break
        attempts += 1
        
        current_L = _get_L(counts, max_check)
        if current_L >= max_check:
            break
        
        target = min(current_L + 50, max_check)
        
        # Recreate counts from current residues
        counts2 = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            _update_counts(counts2, r, p, 1, max_check)
        
        success, new_res, new_cnt = _dpll_backtrack(
            residues, primes, counts2, max_check, deadline, target, max_nodes=10000
        )
        
        if success:
            new_L = _get_L(new_cnt, max_check)
            if new_L > best_L:
                best_residues = new_res.copy()
                best_L = new_L
                # Update working state
                residues = new_res
                counts = new_cnt
            else:
                break
        else:
            # Try smaller target
            target = min(current_L + 10, max_check)
            if target <= current_L:
                break
    
    # Phase 3: More hill climbing from improved state
    for pass_num in range(30):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline and time.time() > deadline:
                return best_residues, best_L
                
            p = primes[i]
            current_r = residues[i]
            
            bottleneck = current_L + 1
            if bottleneck <= max_check:
                target_r = bottleneck % p
            else:
                target_r = current_r
                
            candidates = {current_r, target_r}
            
            if p < 50:
                candidates.update(np.random.randint(0, p, size=10).tolist())
            else:
                candidates.update(np.random.randint(0, p, size=5).tolist())
                
            best_local_r = current_r
            best_local_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                    
                _update_counts(counts, current_r, p, -1, max_check)
                _update_counts(counts, r, p, 1, max_check)
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = r
                else:
                    _update_counts(counts, r, p, -1, max_check)
                    _update_counts(counts, current_r, p, 1, max_check)
                    
            if best_local_r != current_r:
                if best_local_L > current_L:
                    residues[i] = best_local_r
                    current_L = best_local_L
                    improved = True
                    best_residues = residues.copy()
                    best_L = current_L
                else:
                    _update_counts(counts, best_local_r, p, -1, max_check)
                    _update_counts(counts, current_r, p, 1, max_check)
                    
        if not improved:
            break
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Phase 1: Greedy initialization
    residues, counts, current_L = _greedy_construct(primes, max_check)
    best_residues = residues.copy()
    best_L = current_L
    
    # Phase 2: Hill climbing with backtracking
    best_residues, best_L = _hill_climb_with_backtrack(
        best_residues, primes, max_check, deadline
    )
    
    # Phase 3: If time remains, try another round
    if time.time() < deadline - 10:
        best_residues, best_L = _hill_climb_with_backtrack(
            best_residues, primes, max_check, deadline
        )
    
    return best_residues.tolist()

# EVOLVE-BLOCK-END
