import time
import math
import random
import numpy as np

BASE = 51975

def _divisors(n):
    ds = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
    return sorted(ds)

MODULI = [m for m in _divisors(BASE) if m > 1]

# Precompute indices for each modulus and residue? 
# N=51975, sum of m for all divisors is large.
# Better to compute on the fly using modulo operations and bincount.

def score_classes(classes):
    covered = np.zeros(BASE, dtype=np.int8)
    for m, r in classes:
        covered[r::m] = 1
    return BASE - int(np.count_nonzero(covered))

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = random.Random(42)
    
    # Precompute residues array for bincount
    residues_cache = {}
    for m in MODULI:
        residues_cache[m] = np.arange(BASE) % m

    def get_best_residue(m, cnt_other):
        """
        Finds the residue r in [0, m) that maximizes the number of uncovered points
        when added to the system defined by cnt_other.
        cnt_other: coverage count array excluding the class with modulus m.
        """
        # We want to maximize sum(1 for x in range(BASE) if x % m == r and cnt_other[x] == 0)
        # This is equivalent to bincount of residues weighted by (cnt_other == 0)
        weights = (cnt_other == 0).astype(np.int32)
        # np.bincount is very fast
        gains = np.bincount(residues_cache[m], weights=weights, minlength=m)
        return int(np.argmax(gains)), int(gains.max())

    def greedy(order):
        cnt = np.zeros(BASE, dtype=np.int32)
        used = {}
        for m in order:
            # Find best r for this m given current cnt
            best_r, _ = get_best_residue(m, cnt)
            # Add it
            idx = np.arange(best_r, BASE, m)
            cnt[idx] += 1
            used[m] = best_r
        return used

    # Generate a few greedy solutions with different orders
    # Order 1: Small moduli first (cover more points per class)
    # Order 2: Large moduli first
    # Order 3: Random
    
    orders = [
        sorted(MODULI), # Ascending m
        sorted(MODULI, reverse=True), # Descending m
    ]
    for _ in range(4):
        o = MODULI[:]
        rng.shuffle(o)
        orders.append(o)

    best_score = BASE
    best_classes = None
    starts = []
    
    for o in orders:
        used = greedy(o)
        # Calculate score
        cnt = np.zeros(BASE, dtype=np.int32)
        for m, r in used.items():
            idx = np.arange(r, BASE, m)
            cnt[idx] += 1
        covered = int(np.count_nonzero(cnt))
        sc = BASE - covered
        starts.append(used)
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]

    # Local Search
    def local_search(used_init, max_iter):
        used = dict(used_init)
        # Build initial cnt
        cnt = np.zeros(BASE, dtype=np.int32)
        for m, r in used.items():
            idx = np.arange(r, BASE, m)
            cnt[idx] += 1
        
        covered = int(np.count_nonzero(cnt))
        best_sc = BASE - covered
        best_used = dict(used)
        
        # Precompute index arrays for fast add/remove? 
        # Or just use slicing? Slicing is fast.
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            action = rng.random()
            
            if used:
                if action < 0.7:
                    # Switch residue of an existing modulus
                    m = rng.choice(list(used.keys()))
                    cur_r = used[m]
                    
                    # Remove current class to get cnt_other
                    # We need to be careful with counts.
                    # Instead of modifying cnt and restoring, we can compute gain directly?
                    # No, get_best_residue needs cnt_other.
                    # Let's just copy cnt? N=51975, int32 is ~200KB. Copy is fast.
                    cnt_other = cnt.copy()
                    idx_cur = np.arange(cur_r, BASE, m)
                    cnt_other[idx_cur] -= 1
                    
                    best_r, best_gain = get_best_residue(m, cnt_other)
                    
                    # Check if best_r is different from cur_r
                    if best_r != cur_r:
                        # Calculate gain of current
                        cur_gain = int(np.sum((cnt_other == 0)[np.arange(cur_r, BASE, m)]))
                        # Actually, we just need to compare total coverage.
                        # New coverage = covered - (uncovered by cur_r in cnt_other) + (uncovered by best_r in cnt_other)
                        # Wait, cnt_other is the state WITHOUT m.
                        # Current state coverage: cnt.
                        # New state coverage: cnt_other + indicator(best_r).
                        
                        # Let's just apply the change and calculate new covered.
                        # This is O(N) per step. With 10k steps, 500M ops. Might be tight in Python/NumPy.
                        # But NumPy is C-speed. 51975 * 10000 = 5e8. ~0.5-1s per 1e8? So ~5s.
                        # We have 145s. So 10k steps is feasible.
                        
                        # Apply change
                        cnt[idx_cur] -= 1 # Remove old
                        idx_new = np.arange(best_r, BASE, m)
                        cnt[idx_new] += 1 # Add new
                        used[m] = best_r
                        covered = int(np.count_nonzero(cnt))
                        
                        sc = BASE - covered
                        if sc < best_sc:
                            best_sc = sc
                            best_used = dict(used)
                    # If best_r == cur_r, no change needed.
                    # Note: get_best_residue might return cur_r if it's already optimal.

                elif action < 0.9:
                    # Add a new modulus
                    unused = [m for m in MODULI if m not in used]
                    if not unused:
                        continue
                    m = rng.choice(unused)
                    best_r, _ = get_best_residue(m, cnt)
                    
                    # Check if it actually adds coverage
                    # gain = number of x in class(m, best_r) where cnt[x] == 0
                    idx = np.arange(best_r, BASE, m)
                    gain = int(np.sum(cnt[idx] == 0))
                    
                    if gain > 0:
                        cnt[idx] += 1
                        used[m] = best_r
                        covered += gain
                        sc = BASE - covered
                        if sc < best_sc:
                            best_sc = sc
                            best_used = dict(used)
                    # If gain == 0, adding it doesn't help, skip (or add and remove later?)
                    # Usually better to keep the set minimal if no gain.

                else:
                    # Remove a modulus if it's redundant
                    m = rng.choice(list(used.keys()))
                    r = used[m]
                    idx = np.arange(r, BASE, m)
                    
                    # Check if any point in this class is covered ONLY by this class
                    # i.e., cnt[x] == 1 for some x in idx
                    is_redundant = np.all(cnt[idx] > 1)
                    
                    if is_redundant:
                        cnt[idx] -= 1
                        del used[m]
                        # Recalculate covered? 
                        # Since it was redundant, covered count doesn't change.
                        # But let's be safe.
                        # Actually, if cnt[x] > 1 for all x in idx, then removing it doesn't uncover anything.
                        # So covered remains same.
                        # sc = BASE - covered (unchanged)
                        # We might want to remove to allow other moduli to be added?
                        # Or just to simplify.
                        # Let's keep it removed.
                        
            # If not used, we start empty, but greedy always fills it.
            
        return best_used, best_sc

    # Run local search from each start
    for i, used0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # Increase iterations as time allows
        # Estimate time per iteration
        # Let's just run a fixed high number and break on deadline
        used, sc = local_search(used0, max_iter=5000)
        
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]

    # Final polish: try to add any remaining moduli that might help?
    # Or remove any that don't help?
    # The local search already does this.
    
    if best_classes is None:
        # Fallback
        used = greedy(sorted(MODULI))
        best_classes = [[m, used[m]] for m in used]
        
    return best_classes
