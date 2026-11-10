import time
import math
import random

MAXM = 128
DEADLINE = 85.0  # Safety margin before the 90s kill


def _gcd_table():
    g = [[0] * (MAXM + 1) for _ in range(MAXM + 1)]
    for a in range(2, MAXM + 1):
        for b in range(2, MAXM + 1):
            g[a][b] = math.gcd(a, b)
    return g


def _num_divisors(n):
    count = 0
    i = 2
    while i * i <= n:
        if n % i == 0:
            count += 2
            if i * i == n:
                count -= 1
        i += 1
    return count + 1


def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _is_disjoint(r1, m1, r2, m2, gt):
    # Disjoint if r1 != r2 mod gcd(m1, m2)
    return (r1 - r2) % gt[m1][m2] != 0


def _can_add(r, m, selected, gt):
    for (r2, m2) in selected:
        if not _is_disjoint(r, m, r2, m2, gt):
            return False
    return True


def _find_valid_residue(m, selected, gt):
    """Find any valid residue for m given selected classes."""
    for r in range(m):
        if _can_add(r, m, selected, gt):
            return r
    return None


def _greedy(order, gt, time_limit):
    selected = []
    used = set()
    t0 = time.time()
    for m in order:
        if time.time() - t0 > time_limit:
            break
        if m in used:
            continue
        r = _find_valid_residue(m, selected, gt)
        if r is not None:
            selected.append((r, m))
            used.add(m)
    return selected


def _local_search(selected, gt, time_limit):
    """
    Local search: 
    1. Try to add any missing modulus.
    2. Try to replace one modulus with two others.
    """
    t0 = time.time()
    improved = True
    current = [tuple(x) for x in selected]
    
    while improved and (time.time() - t0 < time_limit):
        improved = False
        current_used = set(m for _, m in current)
        
        # Phase 1: Try to add any missing modulus
        for m in range(2, MAXM + 1):
            if m in current_used:
                continue
            r = _find_valid_residue(m, current, gt)
            if r is not None:
                current.append((r, m))
                current_used.add(m)
                improved = True
                # Restart phase 1 check since adding one might allow others
                break 
        
        if improved:
            continue
            
        # Phase 2: Try to replace one with two
        n = len(current)
        for drop_i in range(n):
            if time.time() - t0 > time_limit:
                break
            r_d, m_d = current[drop_i]
            base = [x for i, x in enumerate(current) if i != drop_i]
            base_used = set(m for _, m in base)
            
            # Candidate moduli to add
            cand_mods = [m for m in range(2, MAXM + 1) if m not in base_used]
            
            found_pair = None
            # We need to find two moduli m1, m2 and residues r1, r2 such that:
            # 1. (r1, m1) is valid against base
            # 2. (r2, m2) is valid against base + (r1, m1)
            
            # To speed up, we can iterate through candidates and check compatibility
            # This is O(N^2 * K) where N is num moduli, K is num residues
            # N=128, K=128. 128^2 * 128 ~ 2M ops per drop. 128 drops -> 256M ops.
            # This might be too slow in Python if done naively for many iterations.
            # We will limit the depth or use a heuristic.
            
            # Optimization: Only try a subset of swaps or break early
            # Let's try to find ANY valid pair.
            for m1 in cand_mods:
                if found_pair:
                    break
                r1 = _find_valid_residue(m1, base, gt)
                if r1 is None:
                    continue
                
                base_temp = base + [(r1, m1)]
                
                for m2 in cand_mods:
                    if m2 == m1 or m2 in base_used:
                        continue
                    r2 = _find_valid_residue(m2, base_temp, gt)
                    if r2 is not None:
                        found_pair = [(r1, m1), (r2, m2)]
                        break
            
            if found_pair:
                new_current = base + found_pair
                if len(new_current) > len(current):
                    current = new_current
                    improved = True
                    break # Restart loop with new state
    
    return [list(x) for x in current]


def _seed_solution():
    """
    Start with a strong seed: use moduli that are multiples of small primes
    to maximize the chance of finding valid residues.
    The key insight from the cell findings is that using multiples of a prime
    (like 11) with distinct residues mod that prime guarantees disjointness.
    """
    # Start with powers of 2 (always disjoint if residues are 1)
    # 2, 4, 8, 16, 32, 64, 128
    seed = [(1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (1, 64), (1, 128)]
    return seed


def construct():
    start = time.time()
    deadline = start + DEADLINE
    gt = _gcd_table()
    
    all_mods = list(range(2, MAXM + 1))
    
    # Pre-compute divisor counts for sorting
    div_counts = {m: _num_divisors(m) for m in all_mods}
    primes = [m for m in all_mods if _is_prime(m)]
    composites = [m for m in all_mods if not _is_prime(m)]
    
    # Define several orders
    orders = [
        sorted(all_mods, key=lambda m: div_counts[m], reverse=True), # Most constrained first
        sorted(all_mods, key=lambda m: div_counts[m]), # Least constrained first
        sorted(all_mods, reverse=True), # Descending value
        sorted(all_mods), # Ascending value
        primes + composites, # Primes first
        composites + primes, # Composites first
        sorted(all_mods, key=lambda m: m.bit_count(), reverse=True),
        sorted(all_mods, key=lambda m: m.bit_count()),
    ]
    
    rnd = random.Random(42)
    for _ in range(15):
        o = list(all_mods)
        rnd.shuffle(o)
        orders.append(o)
        
    best = _seed_solution()
    best_size = len(best)
    
    for order in orders:
        if time.time() > deadline - 2.0: # Leave 2s for final local search
            break
            
        # Greedy phase
        remaining_for_greedy = min(1.5, (deadline - time.time()) * 0.2)
        if remaining_for_greedy < 0.1:
            break
        sel = _greedy(order, gt, remaining_for_greedy)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best = [list(x) for x in sel]
        
        # Local Search phase
        remaining_for_ls = deadline - time.time() - 1.0
        if remaining_for_ls < 0.5:
            continue
            
        # Allocate time for local search, ensuring we don't run out
        ls_time = min(remaining_for_ls, 8.0) # Cap at 8s per LS to allow multiple tries if fast
        if ls_time > 0.5:
            sel_ls = _local_search(sel, gt, ls_time)
            if len(sel_ls) > len(sel):
                sel = sel_ls
                if len(sel) > best_size:
                    best_size = len(sel)
                    best = [list(x) for x in sel]
            
    # Final polish on the absolute best
    if time.time() < deadline - 0.5 and best:
        remaining = deadline - time.time() - 0.2
        if remaining > 1.0:
            final_sel = _local_search(best, gt, remaining)
            if len(final_sel) > best_size:
                best = final_sel
                best_size = len(final_sel)

    if not best:
        best = [[1, 2]]
        
    return best

if __name__ == "__main__":
    sol = construct()
    print("size:", len(sol))
    print(sol)
    # Verification
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            r1, m1 = sol[i]
            r2, m2 = sol[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, (i, j, r1, m1, r2, m2)
    mods = [m for (_, m) in sol]
    assert len(set(mods)) == len(mods)
    for r, m in sol:
        assert 0 <= r < m and 2 <= m <= 128
    print("OK")
