import time
def construct():
    deadline = time.time() + 60.0  # safety margin
    result = [1 << i for i in range(12)]
    # Verify validity (all subset sums distinct)
    sums = set()
    for mask in range(1 << 12):
        s = 0
        for i in range(12):
            if mask & (1 << i):
                s += result[i]
        if s in sums:
            # Should never happen, but return seed as fallback
            return [1 << i for i in range(12)]
        sums.add(s)
    return result
