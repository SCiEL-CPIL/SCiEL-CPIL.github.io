import zlib

def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    x = x ^ (x >> 31)
    return x

def priority(vector, n):
    # Convert vector to integer
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    
    # Mix
    h = splitmix64(val)
    
    # Return float in [0, 1)
    return (h % 1000000) / 1000000.0
