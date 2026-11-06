"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5


def _build_table():
    """Map each non-zero quadratic residue to one witness root in [1, P-1]."""
    root = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root:
            root[r] = x
    return root


def _build_bitset():
    """Build a bitset integer where bit i is set iff i is a non-zero QR mod P."""
    bs = 0
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            bs |= (1 << r)
    return bs


def _verify(result):
    s = result['start']
    roots = result['roots']
    if not (1 <= s <= _P - _M):
        return False
    if len(roots) != _M:
        return False
    for i in range(_M):
        v = s + i
        if v == 0 or v >= _P:
            return False
        ri = roots[i]
        if not (1 <= ri <= _P - 1):
            return False
        if (ri * ri) % _P != v:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    # Build the quadratic residue table: residue -> smallest root
    roots = _build_table()

    # Build bitset for fast consecutive-residue check
    bs = _build_bitset()
    mask = (1 << _M) - 1  # 0b11111 for M=5

    # Direct ascending search using bitwise check
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            break
        # Check if bits s through s+4 are all set
        if (bs >> s) & mask == mask:
            result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    # Fallback: backward search (shouldn't happen for this fixed instance)
    for s in range(_P - _M, 0, -1):
        if time.time() > deadline:
            break
        if (bs >> s) & mask == mask:
            result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    # Final fallback: dictionary-based search
    for s in range(1, _P - _M + 1):
        ok = True
        for i in range(_M):
            v = s + i
            if v == 0 or v >= _P or v not in roots:
                ok = False
                break
        if ok:
            result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    raise RuntimeError('No run found')
