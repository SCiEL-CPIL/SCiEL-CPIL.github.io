import struct

# EVOLVE-BLOCK-START
# Hybrid Frequency Estimator: Exact 8-bit counters + Count-Min Sketch overflow correction.
# 
# Structure:
# - Bytes 0-399: Exact 8-bit counters for each item (0..399).
# - Bytes 400-1023: Count-Min Sketch with 4 rows of 78 16-bit counters.
#
# Logic:
# - update: Increment exact counter (saturate at 255). Increment CMS counters.
# - query: If exact counter < 255, return it (exact).
#          Else, return the minimum of the CMS rows (handles high frequencies).
#
# This approach provides zero error for low-frequency items (most items in Uniform and 
# Zipf distributions) and high accuracy for high-frequency items via the CMS.

PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641
]
MOD = 2147483647
EXACT_SIZE = 400
CMS_ROWS = 4
CMS_WIDTH = 78
CMS_BYTES_PER_ROW = CMS_WIDTH * 2
CMS_TOTAL_BYTES = CMS_ROWS * CMS_BYTES_PER_ROW
TOTAL_BYTES = EXACT_SIZE + CMS_TOTAL_BYTES

def update(state, item):
    b = bytearray(state)
    
    # Update exact counter (saturate at 255)
    val = b[item]
    if val < 255:
        b[item] = val + 1
    
    # Update CMS counters
    for i in range(CMS_ROWS):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % CMS_WIDTH
        offset = EXACT_SIZE + i * CMS_BYTES_PER_ROW + idx * 2
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val > 65535:
            val = 65535
        struct.pack_into('<H', b, offset, val)
        
    return bytes(b)

def query(state, item):
    exact_val = state[item]
    if exact_val < 255:
        return exact_val
    else:
        min_val = 65535
        for i in range(CMS_ROWS):
            p = PRIMES[i]
            idx = ((item * p + 31 * i) % MOD) % CMS_WIDTH
            offset = EXACT_SIZE + i * CMS_BYTES_PER_ROW + idx * 2
            val = struct.unpack_from('<H', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val
# EVOLVE-BLOCK-END
