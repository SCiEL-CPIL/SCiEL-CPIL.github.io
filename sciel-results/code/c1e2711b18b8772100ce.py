"""Odd covering systems with moduli dividing 51975."""
import time
import random

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Generate all divisors > 1
def _get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

_DIVS = _get_divisors(BASE)
# Sort divisors: we want to try larger moduli first in some passes, smaller in others
_DIVS_SORTED = sorted(_DIVS)
_DIVS_REVERSE = sorted(_DIVS, reverse=True)

# Precompute bitsets for each (m, r) pair using Python ints (arbitrary precision)
# Each bitset has BASE bits, where bit i is 1 if i % m == r
# We'll use a list of ints indexed by (m_idx, r)

def _build_bitsets():
    """Build bitset representations for all (m, r) pairs."""
    bitsets = {}
    for m in _DIVS:
        for r in range(m):
            bits = 0
            # Set bit i for all i where i % m == r
            # i = r, r+m, r+2m, ...
            i = r
            while i < BASE:
                bits |= (1 << i)
                i += m
            bitsets[(m, r)] = bits
    return bitsets

_BITSETS = None

def _get_bitsets():
    global _BITSETS
    if _BITSETS is None:
        _BITSETS = _build_bitsets()
    return _BITSETS

def _score(classes, covered_bits):
    """Return number of uncovered residues."""
    # Count bits in covered_bits that are 0
    # Total bits = BASE
    # Uncovered = BASE - popcount(covered_bits)
    uncovered = BASE - bin(covered_bits).count('1')
    return uncovered

def _greedy_construct(moduli_order, deadline, random_seed=None):
    """
    Greedy construction: for each modulus in moduli_order, pick the residue
    that covers the most uncovered residues.
    Returns (classes, covered_bits, score).
    """
    bitsets = _get_bitsets()
    covered = 0
    classes = []
    used_moduli = set()
    
    for m in moduli_order:
        if m in used_moduli:
            continue
        if time.time() >= deadline:
            break
        used_moduli.add(m)
        
        # Find best residue for this modulus
        best_r = 0
        best_gain = -1
        for r in range(m):
            bits = bitsets[(m, r)]
            gain = bin(bits & ~covered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_r = r
            if gain == 0:
                continue
        classes.append([m, best_r])
        covered |= bitsets[(m, best_r)]
    
    score = BASE - bin(covered).count('1')
    return classes, covered, score

def _local_search(classes, covered_bits, deadline, max_iters=50):
    """
    Local search: try to swap a class's residue to improve coverage.
    Returns (classes, covered_bits, score).
    """
    bitsets = _get_bitsets()
    score = BASE - bin(covered_bits).count('1')
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        improved = False
        for i, (m, r) in enumerate(classes):
            if time.time() >= deadline:
                break
            # Try all other residues for this modulus
            old_bits = bitsets[(m, r)]
            for r2 in range(m):
                if r2 == r:
                    continue
                new_bits = bitsets[(m, r2)]
                # New covered = (covered - old_bits) | new_bits
                # But we need to be careful: other classes may also cover those bits
                # Actually, covered = OR of all class bitsets
                # If we change class i from r to r2:
                # new_covered = (covered & ~old_bits) | new_bits
                # But this is only correct if no other class covers any bit in old_bits
                # That's not guaranteed. So we need to recompute.
                # For efficiency, let's compute the marginal effect:
                # Bits that were only covered by class i: old_bits & ~(covered ^ old_bits)
                # Actually: covered_without_i = covered & ~old_bits is wrong because
                # other classes might cover those bits too.
                
                # Let's just compute: new_covered = covered | new_bits, but we need
                # to remove the contribution of the old class.
                # This is complex. Let's use a simpler approach:
                # Compute what's covered if we remove class i and add new class i.
                # For now, skip local search if it's too complex.
                pass
        
        # Alternative: try adding a new class or removing a redundant one
        # Try removing a class if it doesn't reduce coverage
        for i in range(len(classes) - 1, -1, -1):
            if time.time() >= deadline:
                break
            m, r = classes[i]
            old_bits = bitsets[(m, r)]
            # Check if removing this class reduces coverage
            # new_covered = covered & ~old_bits ... but again, other classes may cover
            # Let's compute: bits that are covered ONLY by this class
            # = old_bits & ~(OR of all other classes)
            # This is expensive. Skip for now.
            pass
        
        if not improved:
            break
    
    return classes, covered_bits, score

def _multi_pass_greedy(deadline):
    """
    Run greedy with different orderings and return the best result.
    """
    bitsets = _get_bitsets()
    best_classes = None
    best_covered = 0
    best_score = BASE
    
    # Different orderings to try
    orderings = [
        _DIVS_SORTED,        # ascending
        _DIVS_REVERSE,       # descending
        list(reversed(_DIVS_SORTED)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(5):
        shuffled = _DIVS_SORTED[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() >= deadline:
            break
        classes, covered, score = _greedy_construct(order, deadline)
        if score < best_score:
            best_score = score
            best_classes = classes
            best_covered = covered
    
    return best_classes, best_covered, best_score

def _refine_residues(classes, covered_bits, deadline):
    """
    For each class, try all residues and pick the best one.
    This is a simple local improvement.
    """
    bitsets = _get_bitsets()
    
    # First, compute the coverage without each class to understand marginal contributions
    # For simplicity, iterate: for each class, try all residues and pick best
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(classes)):
            if time.time() >= deadline:
                break
            m, r = classes[i]
            # Compute coverage without this class
            # We need to OR all other classes
            other_covered = 0
            for j in range(len(classes)):
                if j != i:
                    other_covered |= bitsets[(classes[j][0], classes[j][1])]
            
            # Now try all residues for class i
            best_r = r
            best_new_covered = 0
            best_score = BASE - bin(other_covered | bitsets[(m, r)]).count('1')
            
            for r2 in range(m):
                if r2 == r:
                    continue
                new_covered = other_covered | bitsets[(m, r2)]
                new_score = BASE - bin(new_covered).count('1')
                if new_score < best_score:
                    best_score = new_score
                    best_new_covered = new_covered
                    best_r = r2
            
            if best_r != r:
                classes[i][1] = best_r
                changed = True
    
    final_covered = 0
    for m, r in classes:
        final_covered |= bitsets[(m, r)]
    final_score = BASE - bin(final_covered).count('1')
    return classes, final_covered, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    # Build bitsets if not already done
    _get_bitsets()
    
    # Multi-pass greedy
    classes, covered, score = _multi_pass_greedy(deadline)
    
    if classes is None:
        # Fallback: simple greedy
        classes, covered, score = _greedy_construct(_DIVS_SORTED, deadline)
    
    # Refine residues if time permits
    if time.time() < deadline - 5.0:
        classes, covered, score = _refine_residues(classes, covered, deadline)
    
    # Final check
    if time.time() >= deadline:
        pass  # Just return what we have
    
    return classes
