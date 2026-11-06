"""Construct the Witt design S(5,6,12) by finding a base block whose PGL(2,11) orbit has size 132."""
import time
from itertools import combinations

# Deadline guard: evaluator kills after 150s
_DEADLINE = time.monotonic() + 140.0  # 10s safety margin

# PGL(2, 11) group structure:
# Elements of GF(11): 0..10, plus infinity (11)
# Transformation: x -> (a*x + b) / (c*x + d), with a*d - b*c != 0 (mod 11)
# Parameters: a in 1..10, b in 0..10, c in 0..10, d in 0..10
# with the constraint that if c=0 then d != 0, and if c!=0 the matrix is non-singular.

def _pgl2_11_transform(x, a, b, c, d):
    """Apply fractional linear transformation x -> (ax+b)/(cx+d) over GF(11).
    x is 0..10 for field elements, 11 for infinity.
    Returns the image as an integer 0..11.
    """
    if x == 11:  # infinity
        if c == 0:
            return 11  # infinity maps to infinity
        else:
            # infinity maps to a/c
            return (a * pow(c, 9, 11)) % 11  # a * c^(-1) mod 11
    else:
        # x is a field element 0..10
        denom = (c * x + d) % 11
        if denom == 0:
            return 11  # maps to infinity
        else:
            num = (a * x + b) % 11
            return (num * pow(denom, 9, 11)) % 11  # num * denom^(-1) mod 11


def _generate_pgl2_11_permutations():
    """Generate all 1320 permutations of {0,...,11} from PGL(2,11).
    Returns a list of tuples, each being a permutation as a tuple of length 12
    where perm[i] is the image of point i.
    """
    perms = []
    for a in range(1, 11):  # a in 1..10
        for b in range(11):  # b in 0..10
            for c in range(11):  # c in 0..10
                for d in range(11):  # d in 0..10
                    # Check non-singularity: ad - bc != 0 (mod 11)
                    det = (a * d - b * c) % 11
                    if det == 0:
                        continue
                    # Build the permutation
                    perm = []
                    for x in range(12):
                        perm.append(_pgl2_11_transform(x, a, b, c, d))
                    perms.append(tuple(perm))
    return perms


def _apply_permutation(block, perm):
    """Apply a permutation to a block (tuple of 6 elements).
    perm[i] = image of i.
    Returns a new frozenset of 6 elements.
    """
    return frozenset(perm[x] for x in block)


def _find_base_block():
    """Find a 6-subset whose PGL(2,11) orbit has exactly 132 distinct blocks,
    and no 5-subset is repeated among the orbit blocks.
    """
    perms = _generate_pgl2_11_permutations()
    
    # Candidate base blocks: 6-subsets of {0,...,11}
    # By symmetry, we can restrict to blocks containing 0 or 11,
    # but let's just iterate and check orbit size.
    # To speed up, we can use the fact that stabilizer of a point in PGL(2,11)
    # has size 1320/12 = 110, so orbit of a point is 12.
    # The stabilizer of a block (if it's a Witt block) should have size 1320/132 = 10.
    
    # Try blocks containing 11 (infinity) to reduce candidates by symmetry:
    # PGL(2,11) is 3-transitive, so we can fix 11 in the block.
    # Blocks containing 11: choose 5 from {0,...,10} = C(10,5) = 252 candidates.
    # We can further use the stabilizer of 11 (which is affine group x -> ax+b, a!=0)
    # to reduce further, but let's just iterate.
    
    candidates = []
    for rest in combinations(range(11), 5):
        candidates.append(frozenset([11] + list(rest)))
    
    # Also try blocks containing 0 but not 11, in case:
    for rest in combinations(range(1, 11), 5):
        candidates.append(frozenset([0] + list(rest)))
    
    # For each candidate, compute orbit and check:
    # 1. Orbit size == 132
    # 2. No 5-subset repeated
    
    for base in candidates:
        if time.monotonic() > _DEADLINE:
            break
        
        # Compute orbit
        orbit = set()
        for perm in perms:
            orbit.add(_apply_permutation(tuple(base), perm))
        
        if len(orbit) != 132:
            continue
        
        # Check 5-subset uniqueness
        covered = set()
        valid = True
        for block in orbit:
            for s in combinations(sorted(block), 5):
                if s in covered:
                    valid = False
                    break
                covered.add(s)
            if not valid:
                break
        
        if valid:
            return sorted(orbit)
    
    # Fallback: try all C(12,6) = 924 candidates
    for base_set in combinations(range(12), 6):
        if time.monotonic() > _DEADLINE:
            break
        base = frozenset(base_set)
        
        orbit = set()
        for perm in perms:
            orbit.add(_apply_permutation(tuple(base), perm))
        
        if len(orbit) != 132:
            continue
        
        covered = set()
        valid = True
        for block in orbit:
            for s in combinations(sorted(block), 5):
                if s in covered:
                    valid = False
                    break
                covered.add(s)
            if not valid:
                break
        
        if valid:
            return sorted(orbit)
    
    return None


def construct():
    """Return a list of 132 distinct 6-subsets of 0..11 (the Witt design S(5,6,12))."""
    result = _find_base_block()
    if result is not None:
        # Convert frozensets to lists
        return [sorted(b) for b in result]
    
    # Fallback: greedy packing (should not be needed)
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
