import hashlib

def priority(vector, n):
    # Convert vector to bytes for hashing
    # Each element is 0, 1, or 2. We can just pack them.
    # Using bytes is efficient.
    b = bytes(vector)
    # Use a strong hash to ensure good distribution
    # sha256 gives 256 bits, which is plenty.
    h = hashlib.sha256(b).digest()
    # Convert the first 8 bytes to an integer for a float in [0, 1)
    # Or use the whole thing?
    # int.from_bytes(h[:8], 'big') / 2**64
    val = int.from_bytes(h[:8], 'big')
    return val / 2**64
