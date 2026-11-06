import zlib

def priority(vector, n):
    # Convert vector to bytes for hashing
    # Vector is a tuple of ints 0,1,2
    # We can pack it into a string or bytes
    s = ''.join(str(x) for x in vector)
    return zlib.crc32(s.encode('utf-8'))
