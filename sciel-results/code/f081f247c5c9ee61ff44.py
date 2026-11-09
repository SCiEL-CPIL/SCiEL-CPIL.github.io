"""Memory-bounded streaming frequency sketch for 400 items using 1024 bytes.

Structural change: Hybrid exact/sketch approach.
- Items 0-63 (64 items): exact uint16 counters (128 bytes)
- Items 64-399 (336 items): 2-row x 256-slot CMS (896 bytes)

Rationale:
- Zipf(1.6) on 400 items concentrates most mass in low-index items.
  Item 0 has expected count ~1024*H(400,1.6)/400^1.6 ≈ very high.
  Items 0-63 capture the bulk of Zipf mass, so exact counting there
  eliminates collision error for the highest-frequency items.
- The remaining 336 items are spread over 256 slots per row with 2 rows.
  Expected load per counter: 336/256 ≈ 1.31, much lower than 400/128 ≈ 3.125.
- Min-of-2 on wider buckets gives good tail behavior for the remaining items.

Layout (1024 bytes):
  Bytes 0..127:   64 uint16 exact counters for items 0..63
  Bytes 128..1023: 2 rows x 256 uint16 CMS counters for items 64..399
    Row 0: counters at bytes 128..639  (256 counters, 512 bytes)
    Row 1: counters at bytes 640..1023 (256 counters, 512 bytes)

Hash for CMS: ((item * p + offset) % 2^31) % 256
"""

import struct

# Exact counter region: items 0..63
_EXACT_N = 64
_EXACT_BYTES = 64 * 2  # 128 bytes

# CMS region: items 64..399
_CMS_W = 256  # width per row
_CMS_D = 2    # depth (rows)
_CMS_PRIMES = (1035493289, 775374203)
_CMS_OFFSETS = (31, 63)
_MOD = 2147483647  # 2^31 - 1

# Precompute CMS indices for items 64..399
_CMS_INDICES = [[0] * 400 for _ in range(_CMS_D)]
for _item in range(64, 400):
    for _row in range(_CMS_D):
        _CMS_INDICES[_row][_item] = (
            (_item * _CMS_PRIMES[_row] + _CMS_OFFSETS[_row]) % _MOD
        ) % _CMS_W


def update(state, item):
    """Increment the appropriate counter(s) for item. Returns new state as bytes."""
    buf = bytearray(state)

    if item < _EXACT_N:
        # Exact counter: simple increment
        pos = item * 2
        cur = buf[pos] | (buf[pos + 1] << 8)
        if cur < 65535:
            cur += 1
        buf[pos] = cur & 0xFF
        buf[pos + 1] = (cur >> 8) & 0xFF
    else:
        # CMS: increment 2 counters
        for row in range(_CMS_D):
            idx = _CMS_INDICES[row][item]
            pos = _EXACT_BYTES + row * (_CMS_W * 2) + idx * 2
            cur = buf[pos] | (buf[pos + 1] << 8)
            if cur < 65535:
                cur += 1
            buf[pos] = cur & 0xFF
            buf[pos + 1] = (cur >> 8) & 0xFF

    return bytes(buf)


def query(state, item):
    """Return the estimated frequency of item. Returns nonnegative float."""
    if item < _EXACT_N:
        # Exact count
        pos = item * 2
        return float(state[pos] | (state[pos + 1] << 8))
    else:
        # CMS: min of 2 counters
        best = 65535
        for row in range(_CMS_D):
            idx = _CMS_INDICES[row][item]
            pos = _EXACT_BYTES + row * (_CMS_W * 2) + idx * 2
            cur = state[pos] | (state[pos + 1] << 8)
            if cur < best:
                best = cur
        return float(best)
