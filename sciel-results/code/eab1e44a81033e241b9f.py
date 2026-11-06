"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5

def _build_table():
    """Map each non-zero quadratic residue to one witness root in [1, P-1] and build a bitmask."""
    root = {}
    qr_mask = 0
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            qr_mask |= (1 << r)
            if r not in root:
                root[r] = x
    return root, qr_mask

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
    deadline = t0 + 80.0  # Safety margin before 90s kill

    # Build the quadratic residue table and bitmask
    roots_dict, qr_mask = _build_table()

    # We need s, s+1, s+2, s+3, s+4 all to be non-zero quadratic residues mod 1009.
    # qr_mask has bit i set if i is a non-zero QR.
    # We check if bits s through s+4 are all set.
    # (qr_mask >> s) & 0x1F == 0x1F
    
    # Search for the smallest s in [1, 1004]
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            break
        
        # Check if s, s+1, s+2, s+3, s+4 are all QRs using bitmask
        # 0x1F is 0b11111
        if (qr_mask >> s) & 0x1F == 0x1F:
            result = {
                'start': s,
                'roots': [roots_dict[s + i] for i in range(_M)]
            }
            # Verify before returning to be safe
            if _verify(result):
                return result

    # Fallback: if the bitmask check failed for some reason (shouldn't happen), do a direct check
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            break
        if all((s + i) in roots_dict for i in range(_M)):
            result = {
                'start': s,
                'roots': [roots_dict[s + i] for i in range(_M)]
            }
            if _verify(result):
                return result

    raise RuntimeError('No valid run found')
