import struct

# EVOLVE-BLOCK-START
# 16 rows of 64 columns, 8-bit counters.
# Total bytes: 16 * 64 * 1 = 1024.

# Multipliers for hashing.
# We use large odd numbers to ensure good distribution modulo 2^31-1 and then modulo 64.
MULTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    123456789,
    987654321,
    1123581321,
    258195004,
    3141592653 % 2147483647, # ~1594109006
    1414213562,
    1732050808,
    1618033988
]

MOD = 2147483647

def update(state, item):
    s = bytearray(state)
    for i, m in enumerate(MULTS):
        # Calculate index for row i
        # (item * m + salt) % MOD % 64
        # Using salt = i+1 to decorrelate rows slightly, though m is distinct enough
        idx = ((item * m + i + 1) % MOD) % 64
        pos = i * 64 + idx
        val = s[pos] + 1
        s[pos] = val & 0xFF
        
    return bytes(s)

def query(state, item):
    min_val = 255 # Max possible 8-bit value
    for i, m in enumerate(MULTS):
        idx = ((item * m + i + 1) % MOD) % 64
        pos = i * 64 + idx
        val = state[pos]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
