import time, random
from math import gcd

N = 256
NUMS = list(range(2, N + 1))
divs = {n: [d for d in range(2, n + 1) if n % d == 0] for n in NUMS}


def new_state():
    return {'S': set(), 'list': [], 'cd': [0] * (N + 1)}


def do_add(st, x):
    st['S'].add(x); st['list'].append(x)
    for d in divs[x]:
        st['cd'][d] += 1


def do_remove(st, x):
    st['S'].discard(x); st['list'].remove(x)
    for d in divs[x]:
        st['cd'][d] -= 1


def can_add(st, x):
    cd = st['cd']; L = st['list']
    for a in L:
        if cd[a // gcd(a, x)] >= 2:
            return False
    for b in L:
        h = x // gcd(x, b)
        if cd[h] - (1 if b % h == 0 else 0) >= 1:
            return False
    return True


def blocker_list(st, x):
    cd = st['cd']; L = st['list']; out = []
    for a in L:
        d = a // gcd(a, x)
        if cd[d] >= 2:
            out.append(a)
            for c in L:
                if c != a and c % d == 0:
                    out.append(c)
    for b in L:
        h = x // gcd(x, b)
        if cd[h] - (1 if b % h == 0 else 0) >= 1:
            out.append(b)
            for c in L:
                if c != b and c % h == 0:
                    out.append(c)
    return out


def local_search(st, rng, deadline):
    kicks = 0
    while time.monotonic() < deadline:
        excluded = [x for x in NUMS if x not in st['S']]
        rng.shuffle(excluded)
        added = False
        for x in excluded:
            if can_add(st, x):
                do_add(st, x); added = True; break
        if added:
            kicks = 0; continue
        moved = False
        for x in excluded:
            bl = blocker_list(st, x)
            if not bl:
                continue
            rng.shuffle(bl)
            for rem in bl:
                do_remove(st, rem)
                if can_add(st, x):
                    do_add(st, x); moved = True; break
                else:
                    do_add(st, rem)
            if moved:
                break
        if moved:
            continue
        kicked = False
        for x in excluded:
            bl = blocker_list(st, x)
            if not bl:
                continue
            rng.shuffle(bl)
            rem = bl[:3]
            for y in rem:
                do_remove(st, y)
            if can_add(st, x):
                do_add(st, x); kicked = True
            else:
                for y in rem:
                    do_add(st, y)
            if kicked:
                kicks += 1; break
        if kicked:
            if kicks > 60:
                break
            continue
        break
    return st


def greedy(order, deadline):
    st = new_state()
    for x in order:
        if time.monotonic() >= deadline:
            break
        if can_add(st, x):
            do_add(st, x)
    return st


def construct():
    start = time.monotonic()
    deadline = start + 70.0
    primes = [x for x in NUMS if not any(x % d == 0 for d in range(2, int(x ** 0.5) + 1))]
    best = list(primes); best_size = len(primes)

    def upd(st):
        nonlocal best, best_size
        if len(st['S']) > best_size:
            best_size = len(st['S']); best = list(st['S'])

    composites = [n for n in NUMS if len(divs[n]) >= 2]
    r = 0
    while time.monotonic() < deadline - 5.0:
        rng = random.Random(987654 + r)
        t = r % 4
        if t == 0:
            order = NUMS[:]; rng.shuffle(order)
            st = greedy(order, deadline)
        elif t == 1:
            st = new_state()
            for p in primes:
                do_add(st, p)
            for p in rng.sample(primes, rng.randint(2, 5)):
                do_remove(st, p)
        elif t == 2:
            st = new_state()
            for p in primes:
                do_add(st, p)
            x = rng.choice(composites)
            for p in divs[x]:
                if p in st['S']:
                    do_remove(st, p)
        else:
            order = NUMS[:]; rng.shuffle(order)
            st = greedy(order, deadline)
            for p in rng.sample(NUMS, 3):
                if p in st['S']:
                    do_remove(st, p)
        st = local_search(st, rng, deadline - 3.0)
        upd(st)
        r += 1
    return [int(v) for v in best]
