import zlib

def priority(vector, n):
    # Convert vector to a deterministic hash
    # Using CRC32 on the string representation or bytes
    # String rep is slow. Bytes is better.
    # vector elements are 0, 1, 2.
    # Pack into bytes.
    b = bytes(vector)
    return zlib.crc32(b)
