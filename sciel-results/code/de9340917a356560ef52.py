import math
import time
import random
from math import gcd

class LineCounter:
    """Incremental line counter for fast local search."""
    def __init__(self, points):
        self.points = [list(p) for p in points]
        self.n = len(self.points)
        self.line_counts = {}
        self.ordinary_count = 0
        self._rebuild()

    def _line_key(self, i, j):
        x1, y1 = self.points[i]
        x2, y2 = self.points[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def _rebuild(self):
        self.line_counts = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                key = self._line_key(i, j)
                self.line_counts[key] = self.line_counts.get(key, 0) + 1
        self.ordinary_count = sum(1 for c in self.line_counts.values() if c == 1)

    def move_point(self, idx, new_pos):
        """Move point idx to new_pos, update counts incrementally. Returns new ordinary count."""
        # Remove all lines involving idx
        for j in range(self.n):
            if j == idx:
                continue
            key = self._line_key(idx, j)
            old_count = self.line_counts[key]
            if old_count == 1:
                self.ordinary_count -= 1
            self.line_counts[key] -= 1
            if self.line_counts[key] == 0:
                del self.line_counts[key]

        # Move the point
        self.points[idx] = list(new_pos)

        # Add all lines involving idx
        for j in range(self.n):
            if j == idx:
                continue
            key = self._line_key(idx, j)
            old_count = self.line_counts.get(key, 0)
            if old_count == 0:
                self.ordinary_count += 1
            elif old_count == 1:
                self.ordinary_count -= 1
            self.line_counts[key] = old_count + 1

        return self.ordinary_count

    def get_points(self):
        return [list(p) for p in self.points]

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    lc = LineCounter(points)
    return lc.ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_points = None
    best_count = float('inf')

    random.seed(42)

    # Initial configurations
    # 1. 4x3 Grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]

    # 2. 3x4 Grid (Transposed)
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]

    # 3. Points on a parabola y = x^2 (12 points)
    parabola = [[i, i*i] for i in range(12)]

    # 4. Points on a line plus offsets
    def make_config_4():
        pts = []
        # 4 points on x-axis
        for i in range(4):
            pts.append([i, 0])
        # 4 points on y=1
        for i in range(4):
            pts.append([i, 1])
        # 4 points on y=2
        for i in range(4):
            pts.append([i, 2])
        return pts
    config_4 = make_config_4()

    # 5. Grid with shifted columns
    def make_config_5():
        pts = []
        for col in range(4):
            for row in range(3):
                pts.append([col, row + (col % 2)])
        return pts
    config_5 = make_config_5()

    # 6. Random
    def random_points():
        pts = set()
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        return [list(p) for p in pts]

    initial_configs = [grid_4x3, grid_3x4, parabola, config_4, config_5, random_points(), random_points()]

    for init_pts in initial_configs:
        if time.time() > deadline:
            break

        # Verify distinctness and bounds
        if len(set(tuple(p) for p in init_pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in init_pts):
            continue

        lc = LineCounter(init_pts)
        curr_count = lc.ordinary_count

        if curr_count < best_count:
            best_count = curr_count
            best_points = lc.get_points()

        # Simulated Annealing
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05

        while time.time() < deadline:
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = lc.points[idx]

            # Generate a new position
            r = random.random()
            if r < 0.5:
                # Small move
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
            elif r < 0.8:
                # Medium move
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                # Large move
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)

            # Check bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue

            # Check distinctness
            is_distinct = True
            for k in range(12):
                if k != idx and lc.points[k] == [new_x, new_y]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue

            # Try the move
            old_pts = lc.get_points()
            new_count = lc.move_point(idx, [new_x, new_y])

            delta = new_count - curr_count

            # Accept if better or with probability exp(-delta/T)
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < math.exp(-delta / T)

            if accept:
                curr_count = new_count
                if curr_count < best_count:
                    best_count = curr_count
                    best_points = lc.get_points()
            else:
                # Revert
                for i in range(12):
                    lc.points[i] = old_pts[i]
                lc._rebuild()
                curr_count = lc.ordinary_count

            T *= cooling_rate
            if T < min_T:
                T = min_T

    return best_points
