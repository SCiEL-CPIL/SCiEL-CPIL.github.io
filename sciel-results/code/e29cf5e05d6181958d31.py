"""Rewritten solution for smooth two-summand decomposition of 123457."""
import time
import math

def construct():
    deadline = time.time() + 80.0
    TARGET = 123457

    # Compute largest prime factor for all n up to TARGET using a sieve
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_pair = (1, TARGET - 1)

    for a in range(1, TARGET):
        if time.time() > deadline - 1.0:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        val = pa if pa > pb else pb
        if val < best_val:
            best_val = val
            best_pair = (a, b)
            if best_val <= 3:
                break

    a, b = best_pair
    return [a, b]
