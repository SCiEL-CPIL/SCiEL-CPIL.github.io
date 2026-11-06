import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: is_qr[r] = True if r is a non-zero QR mod p
    # roots[r] = smallest x in [1, p-1] with x^2 % p == r
    is_qr = [False] * p
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if r not in roots:
                roots[r] = x

    # Sliding window of length 5 over indices 1..1008
    # Find minimum s in [1, 1004] such that is_qr[s..s+4] are all True
    count = sum(1 for i in range(1, 6) if is_qr[i])

    for s in range(1, 1005):
        if time.time() > deadline:
            # Should never trigger for this fixed-size problem, but guard anyway
            raise RuntimeError('Deadline exceeded')
        if count == 5:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
        # Slide window: remove position s, add position s+5
        if s < 1004:
            if is_qr[s]:
                count -= 1
            nxt = s + 5
            if nxt <= 1008 and is_qr[nxt]:
                count += 1

    raise RuntimeError('No valid run found')
