import sys
sys.setrecursionlimit(100000)

def seed_construct():
    from math import isqrt
    values = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
    return values

s = seed_construct()
print("seed length:", len(s))
print("seed first 20:", s[:20])
print("seed last 20:", s[-20:])
gaps = [s[i+1]-s[i] for i in range(len(s)-1)]
print("seed gaps nondecreasing:", all(gaps[i+1]>=gaps[i] for i in range(len(gaps)-1)))
print("seed max prime:", max(s))
