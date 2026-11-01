"""Rewritten program: quadratic-residue construction plus local polish."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the score for a sorted list U of nonnegative ints containing 0."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return -1.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    n = len(U)
    # Compute sumset and diffset
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return -1.0
    base = 2 * max_u + 1
    if base <= 1:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(base)


def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) for prime p and integer a (0 <= a < p)."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p) == 1


def _quadratic_residues(p):
    """Return sorted list of quadratic residues mod p (including 0)."""
    # QR includes 0 and all x^2 mod p for x in 0..p-1
    qr = set()
    for x in range(p):
        qr.add((x * x) % p)
    return sorted(qr)


def _local_search(U, deadline, max_steps=200):
    """Hill-climb: add, remove, or replace elements to improve score."""
    current = list(U)
    best_score = _score_set(current)
    no_improve = 0
    max_no_improve = 30
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        max_cur = max(current)
        
        # Try adding an element
        if len(current) < 20:
            candidates = set()
            # Near existing elements
            for x in current:
                for dx in range(-3, 4):
                    v = x + dx
                    if v >= 0 and v not in current:
                        candidates.add(v)
            # Beyond max
            for v in range(max_cur + 1, max_cur + 15):
                candidates.add(v)
            # Small values not present
            for v in range(1, min(max_cur + 1, 30)):
                if v not in current:
                    candidates.add(v)
            
            for new_el in sorted(candidates):
                cand = sorted(current + [new_el])
                s = _score_set(cand)
                if s > best_score + 1e-12:
                    best_score = s
                    current = cand
                    max_cur = max(current)
                    improved = True
                    no_improve = 0
                    break
        
        # Try removing an element (not 0)
        if len(current) > 3:
            for i in range(len(current) - 1, 0, -1):  # remove from end first
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = _score_set(cand)
                if s > best_score + 1e-12:
                    best_score = s
                    current = cand
                    max_cur = max(current)
                    improved = True
                    no_improve = 0
                    break
        
        # Try replacing a non-zero element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                orig = current[i]
                # Try nearby values
                for new_el in range(max(1, orig - 8), orig + 9):
                    if new_el == orig or new_el in current:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand or len(cand) < 3:
                        continue
                    s = _score_set(cand)
                    if s > best_score + 1e-12:
                        best_score = s
                        current = cand
                        max_cur = max(current)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
    
    return current, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_score = -1.0
    best_set = [0, 1, 3]
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Quadratic residues for various primes
    # Theoretically, |QR - QR| ~ p and |QR + QR| ~ p/2, ratio ~ 2
    # Score = 1 + log(2)/log(2p+1) which decreases with p
    # So smaller primes give higher scores, but we need enough elements
    
    # Try primes from small to moderately large
    primes_to_try = []
    # Generate primes up to ~500
    for n in range(7, 500):
        if n % 2 == 0 or n % 3 == 0:
            if n > 3:
                continue
        is_prime = True
        if n < 2:
            is_prime = False
        elif n < 4:
            is_prime = True
        else:
            for d in range(2, int(math.sqrt(n)) + 1):
                if n % d == 0:
                    is_prime = False
                    break
        if is_prime:
            primes_to_try.append(n)
    
    # Sort by size ascending (smaller p -> higher score)
    for p in sorted(primes_to_try):
        if time.time() > deadline - 30:
            break
        qr = _quadratic_residues(p)
        # QR always contains 0
        try_set(qr)
        
        # Also try QR plus some extra elements to break symmetry
        # Add a few non-residues to see if it helps
        non_res = [x for x in range(p) if x not in qr]
        if non_res:
            # Add one or two non-residues
            for extra in non_res[:3]:
                cand = sorted(qr + [extra])
                try_set(cand)
            if len(non_res) >= 2:
                cand = sorted(qr + [non_res[0], non_res[1]])
                try_set(cand)
    
    # Phase 2: Small exhaustive search (size 3-6, max_u up to 15)
    for max_u in range(2, 16):
        if time.time() > deadline - 20:
            break
        elements = list(range(1, max_u + 1))
        for size in range(2, 6):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # Phase 3: Structured sets - two blocks, geometric, arithmetic
    for max_u in range(5, 100):
        if time.time() > deadline - 15:
            break
        
        # Two blocks: {0..a} U {b..b+a}
        for a in range(1, min(8, max_u // 2)):
            for b in range(a + 1, max_u - a + 1):
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # Geometric
        for r in [2, 3, 4, 5]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
        
        # Arithmetic progression
        for d in range(1, 8):
            ap = list(range(0, max_u + 1, d))
            try_set(ap)
    
    # Phase 4: Local search from best set found so far
    if time.time() < deadline - 5:
        polished, ps = _local_search(best_set, deadline - 5, max_steps=300)
        if ps > best_score:
            best_score = ps
            best_set = polished
    
    # Phase 5: Local search from a few QR-based starting points
    if time.time() < deadline - 5:
        # Pick a medium prime
        for p in [17, 31, 43, 61, 73]:
            if time.time() > deadline - 5:
                break
            qr = _quadratic_residues(p)
            polished, ps = _local_search(qr, deadline - 5, max_steps=100)
            if ps > best_score:
                best_score = ps
                best_set = polished
    
    return list(best_set)

# EVOLVE-BLOCK-END
