import time
import random
import sys
import math
import itertools
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    deadline = time.time() + 110

    # Fast girth check for circulant graph C_n(a, b)
    # Steps: 0=+a, 1=-a, 2=+b, 3=-b
    # Inverse pairs: (0,1), (1,0), (2,3), (3,2)
    _inverse = (1, 0, 3, 2)

    def _check_short_cycle(n, a, b, L):
        """Check if C_n(a,b) has a non-backtracking cycle of length L"""
        deltas = (a, -a, b, -b)
        if L <= 1:
            return False
        # BFS on (position, last_step)
        # Level 0: start at 0 with no previous step
        # We track visited (pos, step) to avoid re-exploring
        visited = set()
        # Current frontier: list of (pos, last_step)
        current = [(0, -1)]
        for level in range(L):
            next_frontier = []
            for pos, last in current:
                for step in range(4):
                    if last >= 0 and step == _inverse[last]:
                        continue
                    new_pos = (pos + deltas[step]) % n
                    state = (new_pos, step)
                    if state not in visited:
                        visited.add(state)
                        next_frontier.append(state)
            if not next_frontier:
                return False
            current = next_frontier
        # Check if any state in current has pos == 0
        for pos, step in current:
            if pos == 0:
                return True
        return False

    def _girth_at_least(n, a, b, g):
        """Check if C_n(a,b) has girth >= g"""
        for L in range(3, g):
            if _check_short_cycle(n, a, b, L):
                return False
        return True

    # Search for circulant graph for k=4, g=10
    if k == 4 and g == 10:
        found = False

        # Try n values from 320 upward
        for n in range(320, 370):
            if time.time() > deadline:
                break

            for a in range(1, min(n // 2, 12) + 1):
                if time.time() > deadline:
                    break
                if math.gcd(n, a) != 1:
                    continue

                for b in range(a + 1, n // 2 + 1):
                    if time.time() > deadline:
                        break

                    # Connectivity check
                    if math.gcd(math.gcd(n, a), b) != 1:
                        continue

                    if _girth_at_least(n, a, b, g):
                        found = True
                        break

                if found:
                    break

            if found:
                break

        if found:
            edges = []
            for i in range(n):
                edges.append((i, (i + a) % n))
                edges.append((i, (i + b) % n))
            return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
