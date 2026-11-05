# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before the 90 s kill

    # Precompute 2,3,5-smooth numbers up to 128.
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)

    # For each target, the cheapest valid antichain.
    #   1 term: {n} (always valid, n|n is the only pair and there are none).
    #   2 terms: a+b=n, a<b, a does not divide b (a|b is the only possible
    #            divisibility since a<b; a|b <=> b==2a for a+b=n, i.e. n even
    #            and a==n//2).
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Safety margin: fall back to the always-valid 1-term form.
            answers.append([n])
            continue

        best = None
        # 2-term search: iterate a over smooth numbers < n.
        # a|b occurs iff n even and a == n//2 -> skip that single a.
        if n % 2 == 0:
            skip = n // 2
        else:
            skip = -1
        for a in smooth:
            if a >= n:
                break
            if a == skip:
                continue
            b = n - a
            if b <= a:
                continue
            if b in smooth:
                # a<b and a not dividing b (guaranteed by skipping a==n//2)
                best = [a, b]
                break

        if best is None:
            best = [n]
        answers.append(best)

    return answers
# EVOLVE-BLOCK-END
