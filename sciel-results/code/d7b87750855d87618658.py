import time

_P = 1009
_M = 5

def _build_roots_and_bitset():
    """
    Build:
      - roots_map: dict mapping each non-zero QR r to its smallest root x in [1, P-1]
      - bitset: Python int where bit i is 1 iff value (i+1) is a non-zero QR
        (i.e., bit 0 corresponds to value 1, bit 1 to value 2, ..., bit 1007 to value 1008)
    """
    roots_map = {}
    bitset = 0
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            if r not in roots_map:
                roots_map[r] = x
            # Set bit at position (r-1) since bit i corresponds to value i+1
            bitset |= (1 << (r - 1))
    return roots_map, bitset

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    roots_map, bitset = _build_roots_and_bitset()

    # We need to find the smallest s in [1, 1004] such that s, s+1, ..., s+4 are all non-zero QRs.
    # In the bitset, value v corresponds to bit at position (v-1).
    # So values s, s+1, s+2, s+3, s+4 correspond to bits at positions s-1, s, s+1, s+2, s+3.
    # We want all 5 consecutive bits starting at position (s-1) to be set.
    # Create a mask with 5 consecutive 1s: mask = 0b11111 = 31
    # For a given s, check: (bitset >> (s-1)) & 31 == 31
    
    # Iterate s from 1 to 1004
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            # Fallback: shouldn't happen given the tiny search space
            raise RuntimeError("Deadline exceeded during search")
        
        # Check if bits at positions s-1 through s+3 are all set
        if (bitset >> (s - 1)) & 31 == 31:
            # All five values s, s+1, ..., s+4 are non-zero QRs
            roots = [roots_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run found')
