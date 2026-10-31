import zlib

def priority(vector, n):
    # Check if vector is in {0,1}^n
    # Using sum is faster than list comprehension?
    # sum(v for v in vector) < n means no 2s?
    # No, 0s and 1s sum to <= n. 2s increase sum.
    # Actually, just check for 2.
    has_two = 2 in vector
    
    if not has_two:
        # High priority
        # Fast hash
        h = zlib.crc32(bytes(vector))
        # Map to [0.9, 1.0]
        return 0.9 + (h / 2**32) * 0.1
    else:
        # Low priority
        h = zlib.crc32(bytes(vector))
        # Map to [0.0, 0.9]
        return (h / 2**32) * 0.9
