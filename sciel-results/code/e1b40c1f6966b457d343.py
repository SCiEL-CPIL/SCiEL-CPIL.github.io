import hashlib

def priority(vector, n):
    # Convert tuple to bytes for hashing
    # Vector elements are 0, 1, 2
    b = bytes(vector)
    # Use Blake2b for better mixing and distribution
    h = int.from_bytes(hashlib.blake2b(b, digest_size=4).digest(), 'big')
    return float(h)
