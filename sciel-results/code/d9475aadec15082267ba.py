import zlib

def priority(vector, n):
    # Use a stable hash to randomize the order
    # This avoids the lexicographic bias which favors vectors with many 0s
    # Randomized greedy insertion is a known effective heuristic for cap sets
    v_str = ''.join(map(str, vector))
    return float(zlib.crc32(v_str.encode('utf-8')))
