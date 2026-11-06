import time
import random
from bisect import bisect_left, bisect_right

def _valid(S):
    """Check the property for sorted set S. Returns True if valid."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True

def _conflict(a, S):
    """Check if adding a to S would cause a conflict.
    S is assumed to be a set of elements all > a (or we filter).
    Conflict exists if there are b, c in S with b <= c and a | (b+c).
    Since a is the smallest element in the new set, we only need to check
    pairs (b,c) from S where b,c > a. But actually, if a is added,
    a becomes the new minimum (or one of them).
    The condition is: for every a' < b' <= c' in S_new, a' does not divide b'+c'.
    If we add a, and a is the new minimum, then we need to check:
    1. a does not divide b+c for any b<=c in S (where b,c > a? No, b,c in S).
       Wait, if a is the new minimum, then for any pair (b,c) in S,
       if b <= c, we need a NOT to divide b+c.
       Also, we need to ensure that for any existing a' in S, and new a,
       the condition holds. But since a is the new minimum, a < a' for all a' in S.
       So the only new constraints are:
       - a does not divide b+c for any b<=c in S.
       - For any a' in S, a' does not divide a + b for any b in S with a < b (which is all b in S).
         Wait, the condition is a' < b' <= c'.
         If we add a, and a is the smallest, then a' (old min) > a.
         So a is not a' in the condition a' < b' <= c'.
         The new pairs involving a are:
         (a, b, c) where a < b <= c.
         Here a is the first element. So we need: a does not divide b+c for any b<=c in S.
         Are there other pairs?
         What if a is not the smallest?
         In our greedy approach, we add elements in descending order or similar.
         Let's assume S contains elements all > a. Then a is the new minimum.
         The only new triples are those where a is the smallest element.
         i.e., (a, b, c) with a < b <= c.
         So we just need to check: for all b, c in S with b <= c, (b+c) % a != 0.
    """
    lst = sorted(S)
    m = len(lst)
    for j in range(m):
        b = lst[j]
        for k in range(j, m):
            c = lst[k]
            if (b + c) % a == 0:
                return True
    return False

def _conflict_fast(a, S):
    """Faster conflict check using residues."""
    # We need to check if there exist b, c in S (b <= c) such that b + c ≡ 0 (mod a)
    # i.e., c ≡ -b (mod a)
    residues = {}
    for x in S:
        r = x % a
        residues[r] = residues.get(r, 0) + 1
        # Also, if we have seen a residue r' such that (r + r') % a == 0, 
        # we need to ensure that the corresponding elements allow b <= c.
        # Actually, just checking existence of pair is not enough if we care about order?
        # No, if b and c are in S, and b+c is divisible by a, it's a conflict.
        # The condition b <= c is just to avoid double counting or to define the triple order?
        # The triple is (a, b, c) with a < b <= c.
        # If we find any b, c in S such that b+c ≡ 0 mod a, is it always a conflict?
        # Yes, because we can order them as min(b,c) and max(b,c), so let b' = min(b,c), c' = max(b,c).
        # Then b' <= c', and b'+c' = b+c, so a divides b'+c'.
        # So we just need to check if there exist two elements in S (not necessarily distinct? 
        # Wait, b and c are from S. Can b=c? Yes, "Repeated larger summands b=c are included".
        # So if there exists x in S such that 2x ≡ 0 mod a, that's a conflict (with b=c=x).
        # If there exist distinct x, y in S such that x+y ≡ 0 mod a, that's a conflict.
        
    # Let's do a more efficient check:
    # For each b in S, we need to know if there exists c in S with c >= b and (b+c) % a == 0.
    # i.e., c ≡ -b (mod a) and c >= b.
    # Let's group S by residue mod a.
    # For each residue r, let S_r be the sorted list of elements in S with x % a == r.
    # For each b in S, let r_b = b % a. We need c in S_{(-r_b) % a} such that c >= b.
    # We can precompute for each residue the sorted list, and use binary search.
    
    residues = {}
    for x in S:
        r = x % a
        if r not in residues:
            residues[r] = []
        residues[r].append(x)
    
    for r in residues:
        residues[r].sort()
    
    for b in S:
        r_b = b % a
        target_r = (-r_b) % a
        if target_r in residues:
            lst_c = residues[target_r]
            # Find first c in lst_c such that c >= b
            idx = bisect_left(lst_c, b)
            if idx < len(lst_c):
                return True
    return False

def construct():
    start = time.time()
    deadline = start + 75.0
    n = 256
    
    # Start with a known valid set: [171, 256]
    # Why is this valid?
    # For a in [171, 256], b, c in [171, 256] with a < b <= c.
    # Min b+c = 172+172 = 344. Max b+c = 256+256 = 512.
    # a is at least 171.
    # 2a is at least 342. 3a is at least 513.
    # So b+c is in [344, 512].
    # The only multiple of a in this range could be 2a.
    # If b+c = 2a, then a = (b+c)/2.
    # Since b, c >= 172, a >= 172.
    # But we assumed a < b, so a < 172. Contradiction.
    # Wait, a is in [171, 256]. b, c in [171, 256].
    # If a=171, b=172, c=172 -> b+c=344. 344/171 = 2.01... not integer.
    # If a=172, b=173, c=173 -> b+c=346. 346/172 = 2.01...
    # Generally, 2a is even. b+c must be even.
    # If b+c = 2a, then a = (b+c)/2.
    # Since b > a, b >= a+1. c >= b >= a+1.
    # So b+c >= 2a+2 > 2a.
    # So b+c cannot be 2a.
    # Next multiple is 3a.
    # 3a >= 3*171 = 513 > 512.
    # So no multiple of a is in [b+c].
    # Thus [171, 256] is valid.
    
    S = set(range(171, n + 1))
    
    # Try to add elements from 170 down to 1
    # We will use _conflict_fast for speed
    for a in range(170, 0, -1):
        if time.time() > deadline - 10.0:
            break
        if a in S:
            continue
        if not _conflict_fast(a, S):
            S.add(a)
    
    # Local search: try to swap or add/remove
    # We can try to remove an element and see if we can add two or more.
    # Or simply try to add any remaining element if possible (in case order mattered).
    # But we already tried all in descending order.
    # Maybe try ascending?
    # Or random restarts?
    
    # Let's try a few random restarts or different orders if time permits.
    best_S = set(S)
    
    # Try ascending order
    S2 = set(range(171, n + 1))
    for a in range(1, 171):
        if time.time() > deadline - 5.0:
            break
        if a in S2:
            continue
        if not _conflict_fast(a, S2):
            S2.add(a)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Try random order
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            S3 = set(range(171, n + 1))
            candidates = list(range(1, 171))
            random.shuffle(candidates)
            for a in candidates:
                if time.time() > deadline - 5.0:
                    break
                if a in S3:
                    continue
                if not _conflict_fast(a, S3):
                    S3.add(a)
            if len(S3) > len(best_S):
                best_S = S3
    
    # Final verification
    if not _valid(best_S):
        best_S = set(range(171, n + 1))
        
    return list(best_S)
