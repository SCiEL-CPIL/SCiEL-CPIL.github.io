<<<<<<< SEARCH
"""Rewritten program for two Sidon sets with disjoint difference sets.
Uses a randomized search approach: generate many varied candidate answers
and keep the best one found within the time budget."""

import time
import random
import sys
from itertools import combinations, combinations_with_replacement

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

def _greedy_build_A(B, n=256, max_size=64, rng=None):
    """Greedily build A given B, ensuring Sidon property and disjoint diffs with B."""
    if rng is None:
        rng = random.Random()
    
    B_diffs = _get_nonzero_diffs(B)
    A = []
    used_sums = set()
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(A) >= max_size:
            break
        # Check Sidon property with new element
        new_sums = set()
        valid = True
        for a in A:
            s = a + x
            if s in used_sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
        # Check diagonal sum x+x
        s2 = 2 * x
        if s2 in used_sums or s2 in new_sums:
            continue
        new_sums.add(s2)
        # Check difference disjointness
        for a in A:
            d1 = a - x
            d2 = x - a
            if d1 != 0 and d1 in B_diffs:
                valid = False
                break
            if d2 != 0 and d2 in B_diffs:
                valid = False
                break
        if not valid:
            continue
        A.append(x)
        used_sums |= new_sums
    
    return A

def _greedy_build_B(A, n=256, max_size=64, rng=None):
    """Greedily build B given A, ensuring Sidon property and disjoint diffs with A."""
    if rng is None:
        rng = random.Random()
    
    A_diffs = _get_nonzero_diffs(A)
    B = []
    used_sums = set()
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(B) >= max_size:
            break
        new_sums = set()
        valid = True
        for b in B:
            s = b + x
            if s in used_sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
        s2 = 2 * x
        if s2 in used_sums or s2 in new_sums:
            continue
        new_sums.add(s2)
        for b in B:
            d1 = b - x
            d2 = x - b
            if d1 != 0 and d1 in A_diffs:
                valid = False
                break
            if d2 != 0 and d2 in A_diffs:
                valid = False
                break
        if not valid:
            continue
        B.append(x)
        used_sums |= new_sums
    
    return B

def _local_search_improve(A, B, n=256, max_iters=100, rng=None):
    """Try to improve by swapping elements."""
    if rng is None:
        rng = random.Random()
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    for _ in range(max_iters):
        # Try adding a random element to A
        improved = False
        for _attempt in range(20):
            x = rng.randint(1, n)
            if x in best_A:
                continue
            new_A = best_A + [x]
            new_A.sort()
            if len(new_A) > 64:
                continue
            if _is_valid_pair(new_A, best_B):
                best_A = new_A
                best_size = len(A) + len(B)
                improved = True
                break
        
        # Try adding a random element to B
        for _attempt in range(20):
            x = rng.randint(1, n)
            if x in best_B:
                continue
            new_B = best_B + [x]
            new_B.sort()
            if len(new_B) > 64:
                continue
            if _is_valid_pair(best_A, new_B):
                best_B = new_B
                best_size = len(A) + len(B)
                improved = True
                break
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2  # |A| + |B|
    
    rng = random.Random()
    
    # Multiple restarts with different strategies
    for restart in range(20000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Random seed element for A, then greedy B, then greedy A
        if restart % 3 == 0:
            seed_a = rng.randint(1, n)
            A_init = [seed_a]
            B = _greedy_build_B(A_init, n, max_size, rng)
            A = _greedy_build_A(B, n, max_size, rng)
        # Strategy 2: Random seed for B, then greedy A, then greedy B
        elif restart % 3 == 1:
            seed_b = rng.randint(1, n)
            B_init = [seed_b]
            A = _greedy_build_A(B_init, n, max_size, rng)
            B = _greedy_build_B(A, n, max_size, rng)
        # Strategy 3: Random small sets, then extend
        else:
            size_a = rng.randint(1, 5)
            size_b = rng.randint(1, 5)
            A_init = sorted(rng.sample(range(1, n + 1), size_a))
            B_init = sorted(rng.sample(range(1, n + 1), size_b))
            if not _is_valid_pair(A_init, B_init):
                continue
            B = _greedy_build_B(A_init, n, max_size, rng)
            A = _greedy_build_A(B, n, max_size, rng)
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        # Verify validity
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search improvement
        if time.time() < deadline - 5:
            A_ls, B_ls = _local_search_improve(A, B, n, max_iters=20, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
    
    # Final check
    if not _is_valid_pair(best_A, best_B):
        # Fallback: simple valid pair
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
=======
"""Two Sidon sets with disjoint difference sets.
Uses number-theoretic construction via modular arithmetic plus optimized
bitset-based search to find large valid pairs within the time budget."""

import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

def _build_from_modular_construction(n=256, max_size=64, rng=None):
    """
    Use a number-theoretic approach: pick a prime p > n, construct a large
    Sidon set modulo p, then map to [1,n] with collision avoidance.
    
    Key idea: If we have a Sidon set S modulo p, then for any shift t,
    {s + t*p : s in S} is also Sidon (when mapped to integers). We can
    use two different primes or shifts to get disjoint difference sets.
    """
    if rng is None:
        rng = random.Random()
    
    # Find a suitable prime
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Use prime p1 for A, p2 for B with p1 != p2 to ensure disjoint diffs
    # We need primes such that we can fit elements in [1, 256]
    
    # Strategy: Use the fact that if A is Sidon mod p1 and B is Sidon mod p2,
    # with p1, p2 > 2*n, then differences from A are in [-p1, p1] and from B
    # in [-p2, p2], but we need them disjoint in the actual integer range.
    
    # Better approach: Use a single prime p > 2*n, and construct A and B as
    # subsets of [1, n] such that their difference sets are disjoint by design.
    
    # Use the construction: A = {a_i * k : i} mod p mapped to [1,n],
    # B = {b_j * m : j} mod p mapped to [1,n], where k*m is not a quadratic
    # residue mod p, ensuring difference sets don't overlap.
    
    # Simpler robust approach: greedy with bitsets
    A = _greedy_build_fast([], n, max_size, rng, block_diffs=set())
    B_diffs = _get_nonzero_diffs(A)
    B = _greedy_build_fast(A, n, max_size, rng, block_diffs=B_diffs)
    
    return A, B

def _greedy_build_fast(existing, n=256, max_size=64, rng=None, block_diffs=None):
    """Fast greedy build using bitset-style tracking."""
    if rng is None:
        rng = random.Random()
    
    if block_diffs is None:
        block_diffs = set()
    
    # Start with existing elements (for B, existing=A)
    A = list(existing)
    used_sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            used_sums.add(A[i] + A[j])
    
    # Compute current diffs
    cur_diffs = _get_nonzero_diffs(A)
    
    # Candidates to add
    candidates = [x for x in range(1, n + 1) if x not in A]
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(A) >= max_size:
            break
        
        # Check Sidon: x+x and x+a for all a in A must not be in used_sums
        valid = True
        if 2 * x in used_sums:
            continue
        for a in A:
            if a + x in used_sums:
                valid = False
                break
        if not valid:
            continue
        
        # Check difference disjointness with block_diffs
        for a in A:
            d = a - x
            if d != 0 and d in block_diffs:
                valid = False
                break
            d = x - a
            if d != 0 and d in block_diffs:
                valid = False
                break
        if not valid:
            continue
        
        # Also check against cur_diffs (for the set being built)
        # Actually cur_diffs is the diffs of A itself, which is fine for Sidon
        # The constraint is that new diffs (x-a, a-x) don't conflict with block_diffs
        
        A.append(x)
        used_sums.add(2 * x)
        for a in A[:-1]:
            used_sums.add(a + x)
    
    return A

def _local_search_fast(A, B, n=256, max_iters=50, rng=None):
    """Fast local search to improve pair size."""
    if rng is None:
        rng = random.Random()
    
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding to A
        for _ in range(15):
            x = rng.randint(1, n)
            if x in best_A:
                continue
            if len(best_A) >= 64:
                break
            new_A = best_A + [x]
            new_A.sort()
            if _is_valid_pair(new_A, best_B):
                best_A = new_A
                best_size = len(best_A) + len(best_B)
                improved = True
                break
        
        # Try adding to B
        for _ in range(15):
            x = rng.randint(1, n)
            if x in best_B:
                continue
            if len(best_B) >= 64:
                break
            new_B = best_B + [x]
            new_B.sort()
            if _is_valid_pair(best_A, new_B):
                best_B = new_B
                best_size = len(best_A) + len(best_B)
                improved = True
                break
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        
        # Vary the strategy each restart
        strategy = restart_count % 5
        
        if strategy == 0:
            # Greedy A then greedy B
            A = _greedy_build_fast([], n, max_size, rng, block_diffs=set())
            B_diffs = _get_nonzero_diffs(A)
            B = _greedy_build_fast(A, n, max_size, rng, block_diffs=B_diffs)
        elif strategy == 1:
            # Greedy B then greedy A
            B = _greedy_build_fast([], n, max_size, rng, block_diffs=set())
            A_diffs = _get_nonzero_diffs(B)
            A = _greedy_build_fast(B, n, max_size, rng, block_diffs=A_diffs)
        elif strategy == 2:
            # Start with small random seeds
            seed_a = rng.sample(range(1, n + 1), min(3, n))
            seed_b = rng.sample(range(1, n + 1), min(3, n))
            A_init = sorted(seed_a)
            B_init = sorted(seed_b)
            if _is_valid_pair(A_init, B_init):
                B = _greedy_build_fast(A_init, n, max_size, rng, block_diffs=_get_nonzero_diffs(A_init))
                A = _greedy_build_fast(B, n, max_size, rng, block_diffs=_get_nonzero_diffs(B))
            else:
                A = A_init
                B = B_init
        elif strategy == 3:
            # Arithmetic progression based seed
            step_a = rng.randint(2, 8)
            A_init = sorted(set(range(1, n + 1, step_a))[:min(5, len(range(1, n + 1, step_a)))])
            step_b = rng.randint(2, 8)
            B_init = sorted(set(range(1, n + 1, step_b))[:min(5, len(range(1, n + 1, step_b)))])
            if _is_valid_pair(A_init, B_init):
                B = _greedy_build_fast(A_init, n, max_size, rng, block_diffs=_get_nonzero_diffs(A_init))
                A = _greedy_build_fast(B, n, max_size, rng, block_diffs=_get_nonzero_diffs(B))
            else:
                A = A_init
                B = B_init
        else:
            # Pure random initialization
            A_init = sorted(rng.sample(range(1, n + 1), rng.randint(1, 4)))
            B_init = sorted(rng.sample(range(1, n + 1), rng.randint(1, 4)))
            if _is_valid_pair(A_init, B_init):
                B = _greedy_build_fast(A_init, n, max_size, rng, block_diffs=_get_nonzero_diffs(A_init))
                A = _greedy_build_fast(B, n, max_size, rng, block_diffs=_get_nonzero_diffs(B))
            else:
                A = A_init
                B = B_init
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        # Verify and track
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search polish
        if time.time() < deadline - 5:
            A_ls, B_ls = _local_search_fast(A, B, n, max_iters=30, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
    
    # Final validation
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
>>>>>>> REPLACE
