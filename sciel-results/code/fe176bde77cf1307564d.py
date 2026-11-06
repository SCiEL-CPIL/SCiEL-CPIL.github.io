# EVOLVE-BLOCK-START
import time

def _is_sidon(S):
    """Check if S is a Sidon set: all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diff_set(S):
    """Return set of nonzero absolute differences."""
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _quadratic_residues(p):
    """Return sorted list of nonzero quadratic residues mod p, in [1, p-1]."""
    res = set()
    for k in range(1, p):
        res.add((k * k) % p)
    res.discard(0)
    return sorted(res)

def _fix_overlap(A, B):
    """Remove elements from A and B to make difference sets disjoint."""
    diffA = _diff_set(A)
    diffB = _diff_set(B)
    overlap = diffA & diffB
    if not overlap:
        return A, B
    
    # Try removing elements from A first, then B
    A_set = set(A)
    B_set = set(B)
    
    # Greedy: remove elements that cause the most overlap
    for _ in range(100):
        overlap = diffA & diffB
        if not overlap:
            break
        # Find element in A whose removal reduces overlap most
        best_remove_A = None
        best_remove_B = None
        best_gain_A = -1
        best_gain_B = -1
        for a in A:
            new_A = [x for x in A if x != a]
            new_diffA = _diff_set(new_A)
            gain = len((diffA & diffB) - (new_diffA & diffB))
            if gain > best_gain_A:
                best_gain_A = gain
                best_remove_A = a
        for b in B:
            new_B = [x for x in B if x != b]
            new_diffB = _diff_set(new_B)
            gain = len((diffA & diffB) - (diffA & new_diffB))
            if gain > best_gain_B:
                best_gain_B = gain
                best_remove_B = b
        
        if best_gain_A >= best_gain_B and best_remove_A is not None:
            A = [x for x in A if x != best_remove_A]
            diffA = _diff_set(A)
        elif best_remove_B is not None:
            B = [x for x in B if x != best_remove_B]
            diffB = _diff_set(B)
        else:
            break
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Use quadratic residues modulo two primes
    p1 = 113
    p2 = 127
    
    A = _quadratic_residues(p1)
    B = _quadratic_residues(p2)
    
    # Verify both are Sidon
    assert _is_sidon(A), "A is not Sidon"
    assert _is_sidon(B), "B is not Sidon"
    
    # Check difference set disjointness
    diffA = _diff_set(A)
    diffB = _diff_set(B)
    overlap = diffA & diffB
    
    if overlap:
        A, B = _fix_overlap(A, B)
        # Re-verify
        assert _is_sidon(A)
        assert _is_sidon(B)
        diffA = _diff_set(A)
        diffB = _diff_set(B)
        assert not (diffA & diffB), "Difference sets still overlap"
    
    # Try to add more elements if time permits
    elapsed = time.time() - start_time
    if elapsed < 60:
        # Try to add elements to A
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in A:
                continue
            new_A = A + [x]
            if _is_sidon(new_A):
                new_diffA = _diff_set(new_A)
                if not (new_diffA & diffB):
                    A = new_A
                    diffA = new_diffA
        
        # Try to add elements to B
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in B:
                continue
            new_B = B + [x]
            if _is_sidon(new_B):
                new_diffB = _diff_set(new_B)
                if not (diffA & new_diffB):
                    B = new_B
                    diffB = new_diffB
    
    # Final verification
    assert len(A) > 0 and len(B) > 0
    assert len(A) <= 64 and len(B) <= 64
    assert all(1 <= x <= 256 for x in A)
    assert all(1 <= x <= 256 for x in B)
    assert _is_sidon(A)
    assert _is_sidon(B)
    assert not (_diff_set(A) & _diff_set(B))
    
    return [A, B]
# EVOLVE-BLOCK-END
