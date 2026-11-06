import time
import random
import sys

def _check_sidon(S):
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _sidon_diag(base, size, max_n):
    """Construct a Sidon set via base-B diagonal: { i*base + j*base^2 } style.
    We use a simple Sidon construction: for a prime p, set { p*i + i^2 mod p }
    mapped into [1, max_n]. Here we use a known Sidon construction:
    A = { b*i + i^2 : i = 0..k-1 } with b large enough to avoid collisions.
    For b > 2*k, all pair sums are unique (Sidon)."""
    b = 2 * size + 2
    elems = []
    for i in range(size):
        v = b * i + i * i
        if 1 <= v <= max_n:
            elems.append(v)
    return elems

def _golomb_ruler(size, max_n):
    """Construct a Golomb ruler (all pairwise differences unique) using
    the classic construction: { i*(i+1)//2 + i*offset } or use
    the Singer construction / simple quadratic: a_i = i*(i-1)//2 * 2 + i.
    We use a_i = i^2 + i  (which gives differences 2(i-j) + (i-j) = 3(i-j), not unique).
    Better: use the 'Mian-Chowla' style greedy but with a formula:
    a_i = i * (i+1) // 2  gives differences that are triangular numbers, not all unique.
    Use: a_i = i^2  -> differences i^2 - j^2 = (i-j)(i+j), can collide.
    Use the known explicit Golomb ruler: a_i = i * (2*i) = 2i^2? No.
    
    Let's use the simple constructive Golomb ruler: a_i = i * base + i^2 with base > 2*size.
    Differences: (i-j)*base + (i^2 - j^2) = (i-j)*base + (i-j)(i+j) = (i-j)(base + i + j).
    Since base > 2*size > i+j for all i,j < size, the term (base + i + j) is unique per (i,j) pair
    only if (i-j) is unique... not guaranteed.
    
    Safest: use the 'optimal' small Golomb rulers known up to size ~10, and for larger,
    use a_i = i * (2*size) + i*i. Differences = (i-j)*(2*size + i + j).
    Since 2*size + i + j ranges from 2*size+1 to 2*size+2*size-1 = 4*size-1, and i-j ranges
    from -(size-1) to (size-1), the product (i-j)*(2*size+i+j) is unique for each unordered pair
    because if (i-j)*(2s+i+j) = (k-l)*(2s+k+l), with |i-j|,|k-l| < 2s, the factor (2s+i+j)
    is in [2s+1, 4s-1], so the representation is unique (like base-(4s) representation).
    Thus differences are unique -> Golomb ruler -> Sidon."""
    base = 2 * size
    elems = []
    for i in range(size):
        v = i * base + i * i + 1  # +1 to shift into [1, max_n]
        if v <= max_n:
            elems.append(v)
    return elems

def _bitsets_for(S, max_n):
    """Return (sums_bitset, diffs_bitset) as Python ints (bitsets) for set S.
    sums_bitset: bit s is 1 if s is a pair sum (a+a included).
    diffs_bitset: bit d is 1 if d is a nonzero absolute difference."""
    sums = 0
    diffs = 0
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums |= (1 << (S[i] + S[j]))
        for j in range(n):
            if i != j:
                d = abs(S[i] - S[j])
                if d > 0:
                    diffs |= (1 << d)
    return sums, diffs

def _can_add_bitset(x, sums, diffs_other, max_size, len_S):
    """Check if x can be added to a set with current sums bitset, given other set's diffs bitset."""
    if len_S >= max_size:
        return False
    # Check 2x not in sums
    if (sums >> (2 * x)) & 1:
        return False
    # Check x + a not in sums for any a: we need to know if any (x+a) is in sums.
    # This requires knowing the elements, not just sums. So bitset alone isn't enough for sum check.
    # We'll use a hybrid: keep the set for sum checks, use bitset for diff checks.
    return True

def _local_search_bitset(A, B, max_n=256, max_size=64, time_budget=50.0, rng=None):
    """Fast local search using bitsets for diff checks and sets for sum checks."""
    if rng is None:
        rng = random.Random(123)
    
    start = time.time()
    A = list(A)
    B = list(B)
    
    def rebuild(sums_A, diffs_A, sums_B, diffs_B, A, B):
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        return sums_A, diffs_A, sums_B, diffs_B
    
    sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
    
    def try_insert(x, target):
        nonlocal sums_A, diffs_A, sums_B, diffs_B
        if target == 'A':
            if len(A) >= max_size:
                return False
            if (2 * x) in sums_A:
                return False
            for a in A:
                if (x + a) in sums_A:
                    return False
                d = abs(x - a)
                if d > 0 and d in diffs_B:
                    return False
            A.append(x)
            for a in A[:-1]:
                sums_A.add(x + a)
                d = abs(x - a)
                if d > 0:
                    diffs_A.add(d)
            sums_A.add(2 * x)
            return True
        else:
            if len(B) >= max_size:
                return False
            if (2 * x) in sums_B:
                return False
            for b in B:
                if (x + b) in sums_B:
                    return False
                d = abs(x - b)
                if d > 0 and d in diffs_A:
                    return False
            B.append(x)
            for b in B[:-1]:
                sums_B.add(x + b)
                d = abs(x - b)
                if d > 0:
                    diffs_B.add(d)
            sums_B.add(2 * x)
            return True
    
    def try_remove(x, target):
        nonlocal sums_A, diffs_A, sums_B, diffs_B
        if target == 'A':
            A.remove(x)
            # Rebuild
            sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
        else:
            B.remove(x)
            sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
    
    for iteration in range(5000):
        if time.time() - start > time_budget:
            break
        improved = False
        used = set(A) | set(B)
        candidates = [x for x in range(1, max_n + 1) if x not in used]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start > time_budget:
                break
            # Try A then B
            if try_insert(x, 'A'):
                improved = True
                break
            if try_insert(x, 'B'):
                improved = True
                break
        
        # Try swaps: remove one from A, try to add two to B, etc.
        if not improved:
            # Try a single swap: remove x from A, try to add y to B (if y not in B)
            swap_done = False
            A_copy = list(A)
            for idx, x in enumerate(A_copy):
                if time.time() - start > time_budget:
                    break
                # Temporarily remove x from A
                A.remove(x)
                sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
                
                # Try to add some y to B
                used_now = set(A) | set(B)
                cands2 = [y for y in range(1, max_n + 1) if y not in used_now]
                rng.shuffle(cands2)
                found = False
                for y in cands2:
                    if try_insert(y, 'B'):
                        found = True
                        break
                if not found:
                    # Put x back
                    A.append(x)
                    sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
                else:
                    swap_done = True
                    break
            
            if not swap_done:
                # Try B -> A swap
                B_copy = list(B)
                for idx, x in enumerate(B_copy):
                    if time.time() - start > time_budget:
                        break
                    B.remove(x)
                    sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
                    
                    used_now = set(A) | set(B)
                    cands2 = [y for y in range(1, max_n + 1) if y not in used_now]
                    rng.shuffle(cands2)
                    found = False
                    for y in cands2:
                        if try_insert(y, 'A'):
                            found = True
                            break
                    if not found:
                        B.append(x)
                        sums_A, diffs_A, sums_B, diffs_B = rebuild(None, None, None, None, A, B)
                    else:
                        swap_done = True
                        break
                
                if not swap_done:
                    break
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Phase 1: Constructive design
    # Use the base-B Sidon construction for A and a shifted version for B
    # A: { b*i + i^2 : i=0..k-1 } with b = 2k+2
    # B: { b*i + i^2 + offset : i=0..m-1 } with different b or offset to avoid diff collisions
    
    best_A = [1]
    best_B = [3]
    best_score = 2
    
    # Try multiple base parameters
    for k in range(5, 30):
        b = 2 * k + 2
        A_cand = [b * i + i * i + 1 for i in range(k)]
        A_cand = [x for x in A_cand if 1 <= x <= 256]
        if not _check_sidon(A_cand):
            continue
        
        # For B, use a different base and offset to ensure disjoint differences
        for m in range(5, 30):
            b2 = 2 * m + 2
            # Try different offsets to shift B
            for offset in range(1, 20):
                B_cand = [b2 * i + i * i + offset for i in range(m)]
                B_cand = [x for x in B_cand if 1 <= x <= 256]
                if not _check_sidon(B_cand):
                    continue
                dA = _diffs(A_cand)
                dB = _diffs(B_cand)
                if dA & dB:
                    continue
                score = len(A_cand) + len(B_cand)
                if score > best_score:
                    best_score = score
                    best_A = list(A_cand)
                    best_B = list(B_cand)
        
        if time.time() > deadline - 30.0:
            break
    
    # Phase 2: Local search to fill remaining capacity
    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        A_ls, B_ls = _local_search_bitset(best_A, best_B, 256, 64, remaining, random.Random(999))
        valid, score = _validate(A_ls, B_ls)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
