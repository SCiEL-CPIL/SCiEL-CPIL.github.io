N = 17

# EVOLVE-BLOCK-START
def construct():
    return _gen(17)

def _gen(n, base=0):
    if n <= 1:
        return []
    if n == 2:
        return [(base, base+1)]
    n1 = n // 2
    n2 = n - n1
    return _gen(n1, base) + _gen(n2, base+n1) + _merge(n1, n2, base)

def _merge(m, n, base):
    if m == 0 or n == 0:
        return []
    if m == 1 and n == 1:
        return [(base, base+1)]
    total = m + n
    comps = []
    # Merge even-indexed elements
    ea = [base + 2*i for i in range((m+1)//2)]
    eb = [base + m + 2*i for i in range((n+1)//2)]
    eo = [base + 2*i for i in range((total+1)//2)]
    comps += _submerge(ea, eb, eo)
    # Merge odd-indexed elements
    oa = [base + 2*i+1 for i in range(m//2)]
    ob = [base + m + 2*i+1 for i in range(n//2)]
    oo = [base + 2*i+1 for i in range(total//2)]
    comps += _submerge(oa, ob, oo)
    # Final
    for i in range(1, total-1, 2):
        comps.append((base+i, base+i+1))
    return comps

def _submerge(a, b, out):
    """Merge sorted lists a and b into out. All in increasing wire order."""
    na, nb = len(a), len(b)
    total = len(out)
    if total <= 1:
        return []
    if total == 2:
        return [(a[0], b[0])]
    # Recursive odd-even merge on logical indices
    # a[i] is logical input i, b[j] is logical input na+j
    # out[i] is logical output i
    # Use Batcher's merge on logical indices
    lm = _logical_merge(na, nb)
    # Map logical to physical
    result = []
    for i, j in lm:
        # i and j are logical input indices
        wi = a[i] if i < na else b[i-na]
        wj = a[j] if j < na else b[j-na]
        result.append((min(wi,wj), max(wi,wj)))
    return result

def _logical_merge(na, nb):
    if na == 0 or nb == 0:
        return []
    if na == 1 and nb == 1:
        return [(0,1)]
    total = na + nb
    comps = []
    # Even logical indices
    e1 = list(range(0, na, 2))
    e2 = list(range(na, total, 2))
    eo = list(range(0, total, 2))
    comps += _submerge(e1, e2, eo)
    # Odd logical indices
    o1 = list(range(1, na, 2))
    o2 = list(range(na+1, total, 2))
    oo = list(range(1, total, 2))
    comps += _submerge(o1, o2, oo)
    for i in range(1, total-1, 2):
        comps.append((i, i+1))
    return comps
# EVOLVE-BLOCK-END
