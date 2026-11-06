import time

def construct():
    p = 1009
    deadline = time.time() + 50  # safety margin before 60s limit

    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)

    best_start = 1
    best_len = 0

    # Two-pointer / sliding window approach
    # An interval [s, e] is valid iff for every x in [s, e], inv[x] is not in [s, e].
    # As we extend e, if inv[e] is in [s, e], we must push s to inv[e] + 1 (or e+1 if self-inverse).
    s = 1
    for e in range(1, p):
        ie = inv[e]
        if ie == e:
            # Self-inverse: e cannot be in any valid interval
            s = e + 1
        elif ie >= s:
            # inv[e] is in [s, e], so we must exclude it
            s = ie + 1
        # Now [s, e] is valid (if s <= e)
        if s <= e:
            cur_len = e - s + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = s
        if time.time() > deadline:
            break

    return [best_start, best_len]
