import struct
import array

# Primes for 8 rows
PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    2269547791, 3266489917, 668265263, 374761393
]
MOD = 2147483647
WIDTH = 64
DEPTH = 8

def update(state, item):
    # Load state into array of unsigned shorts
    # state is 1024 bytes. 8 rows * 64 cols * 2 bytes/col = 1024 bytes.
    # array('H') expects unsigned short.
    # We need to ensure endianness matches. 'H' is native. 
    # The seed used little-endian uint32. 
    # On x86, native is little-endian. 
    # To be safe and explicit, we can use struct or just assume little-endian 
    # as is standard for these tasks on standard hardware.
    # However, struct.unpack('8H', ...) would be wrong because 8*2=16 bytes.
    # We need 512 unsigned shorts.
    
    # Fast path: 
    # 1. Unpack into a list or array
    # 2. Update 8 specific indices
    # 3. Pack back
    
    # Using array module is efficient.
    # Note: array('H', state) interprets state as bytes.
    # If the system is big-endian, this fails. 
    # Given the seed used little-endian explicitly, and most eval environments are x86, 
    # we assume little-endian. 
    # To be robust, we could use struct.unpack_from in a loop or 
    # struct.unpack('<512H', state).
    
    # struct.unpack('<512H', state) returns a tuple of 512 ints.
    # This is fast enough.
    
    counters = list(struct.unpack('<512H', state))
    
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        pos = i * WIDTH + idx
        counters[pos] += 1
        
    # Pack back
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = struct.unpack('<512H', state)
    
    min_val = float('inf')
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        pos = i * WIDTH + idx
        val = counters[pos]
        if val < min_val:
            min_val = val
            
    return min_val
