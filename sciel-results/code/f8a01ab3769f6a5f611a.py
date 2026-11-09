import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

class LineTracker:
    def __init__(self, pts):
        self.lines = {}
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in self.lines:
                    self.lines[key] = set()
                self.lines[key].add(i)
                self.lines[key].add(j)
        self.score = 0
        self.invalid_count = 0
        for key, indices in self.lines.items():
            c = len(indices)
            if c == 4:
                self.score += 1
            elif c >= 5:
                self.invalid_count += 1

    def remove_point(self, idx):
        delta_score = 0
        delta_invalid = 0
        to_delete = []
        for key, indices in list(self.lines.items()):
            if idx in indices:
                old_c = len(indices)
                if old_c == 4:
                    delta_score -= 1
                elif old_c == 5:
                    delta_invalid -= 1
                    delta_score += 1
                indices.remove(idx)
                if not indices:
                    to_delete.append(key)
        for key in to_delete:
            del self.lines[key]
        self.score += delta_score
        self.invalid_count += delta_invalid
        return delta_score, delta_invalid

    def add_point(self, idx, pts):
        delta_score = 0
        delta_invalid = 0
        p = pts[idx]
        n = len(pts)
        touched_lines = set()
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(p, pts[j])
            if key is None:
                continue
            if key in touched_lines:
                continue
            touched_lines.add(key)
            if key not in self.lines:
                self.lines[key] = set()
            old_c = len(self.lines[key])
            self.lines[key].add(idx)
            if old_c == 3:
                delta_score += 1
            elif old_c == 4:
                delta_score -= 1
                delta_invalid += 1
            # if old_c >= 5, still invalid
        self.score += delta_score
        self.invalid_count += delta_invalid
        return delta_score, delta_invalid

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    tracker = LineTracker(points)
    current_score = tracker.score
    best_points = [p[:] for p in points]
    best_score = current_score

    if current_score < 0:
        return best_points

    def get_eff(tracker):
        if tracker.invalid_count > 0:
            return -1000 - tracker.invalid_count
        return tracker.score

    # Main loop: combine greedy line-completion and SA
    # Key structural idea: prioritize moves that complete a 3-point line to 4 points
    # by moving a point from a "waste" position to a position that joins a 3-line.
    
    T = 3.0
    T_min = 0.1
    cooling_rate = 0.99995
    current_eff = get_eff(tracker)

    # Precompute: for each point, what lines does it belong to?
    # We'll recompute when needed.

    def find_greedy_moves(pts, tracker):
        """Find moves that strictly increase score.
        For each point i and each other point j, consider moving i to a position
        that would create a new 4-line (by joining a 3-line).
        """
        n = len(pts)
        best_move = None
        best_gain = 0

        # Find all 3-point lines
        three_lines = {}  # key -> set of 3 indices
        for key, indices in tracker.lines.items():
            if len(indices) == 3:
                three_lines[key] = set(indices)

        for i in range(n):
            old_x, old_y = pts[i]
            # For each 3-line not containing i, find a position for i that lies on that line
            for key, triple in three_lines.items():
                if i in triple:
                    continue
                a, b, c = key
                # Line: ax + by + c = 0
                # We need an integer point (x,y) on this line, within bounds, not colliding
                # Pick a point near the centroid of the triple to stay local
                xs = [pts[t][0] for t in triple]
                ys = [pts[t][1] for t in triple]
                cx = sum(xs) // 3
                cy = sum(ys) // 3

                # Find integer solutions to a*x + b*y + c = 0 near (cx, cy)
                # Try a few candidates
                found = False
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        x = cx + dx
                        y = cy + dy
                        if abs(x) > 1000 or abs(y) > 1000:
                            continue
                        if a * x + b * y + c != 0:
                            continue
                        # Check collision
                        collision = False
                        for k in range(n):
                            if k != i and pts[k][0] == x and pts[k][1] == y:
                                collision = True
                                break
                        if collision:
                            continue
                        # This is a candidate: move i to (x, y)
                        # But we also need to check that removing i from its old lines
                        # doesn't destroy more 4-lines than we create
                        # For now, just record it and evaluate
                        found = True
                        break
                    if found:
                        break
                if not found:
                    continue

                # Evaluate: simulate the move
                old_pos = [old_x, old_y]
                pts[i] = [x, y]
                tracker.remove_point(i)
                tracker.add_point(i, pts)
                gain = tracker.score - current_score
                # Check if valid
                if tracker.invalid_count == 0:
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, [x, y], old_pos)
                # Revert
                pts[i] = old_pos
                tracker.remove_point(i)
                tracker.add_point(i, pts)

        return best_move

    # Also try: for each pair of 3-lines, find a point that lies on both
    # This would create two 4-lines at once (if the point isn't already on them)

    while time.time() < deadline:
        # Try greedy moves first
        greedy_move = find_greedy_moves(points, tracker)
        if greedy_move:
            i, new_pos, old_pos = greedy_move
            points[i] = new_pos
            tracker.remove_point(i)
            tracker.add_point(i, points)
            current_eff = get_eff(tracker)
            if tracker.score > best_score:
                best_score = tracker.score
                best_points = [p[:] for p in points]
            continue

        # If no greedy move, do SA step
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]

        if random.random() < 0.15:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
        else:
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)

        new_x = old_x + dx
        new_y = old_y + dy

        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue

        collision = False
        for k in range(16):
            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                collision = True
                break
        if collision:
            continue

        points[idx] = [new_x, new_y]
        tracker.remove_point(idx)
        tracker.add_point(idx, points)

        new_eff = get_eff(tracker)
        delta = new_eff - current_eff

        if delta >= 0:
            current_eff = new_eff
            if tracker.score > best_score and tracker.invalid_count == 0:
                best_score = tracker.score
                best_points = [p[:] for p in points]
        else:
            prob = min(1.0, pow(2.71828, delta / max(T, T_min)))
            if random.random() < prob:
                current_eff = new_eff
            else:
                points[idx] = [old_x, old_y]
                tracker.remove_point(idx)
                tracker.add_point(idx, points)

        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart: occasionally reheat
            if random.random() < 0.05:
                T = 2.0

    # Final validation
    final_score = fast_count(best_points)
    if final_score < 0:
        grid_pts, _ = grid_configuration()
        return grid_pts

    return best_points
