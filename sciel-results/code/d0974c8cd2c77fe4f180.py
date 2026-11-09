import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    if container_vol == 0:
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Sort indices by various heuristics
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    min_dim_sorted = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted]
    for _ in range(4):
        random_order = list(range(n))
        random.shuffle(random_order)
        orderings.append(random_order)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def validate_placement(placements):
        if len(placements) != n:
            return False
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            bins[b].append((x, y, z, w, h, d))
        for bin_idx, box_list in bins.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    x1, y1, z1, w1, h1, d1 = box_list[i]
                    x2, y2, z2, w2, h2, d2 = box_list[j]
                    if (x1 < x2 + w2 and x2 < x1 + w1 and
                        y1 < y2 + h2 and y2 < y1 + h1 and
                        z1 < z2 + d2 and z2 < z1 + d1):
                        return False
        return True

    class Bin:
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            # Prisms are (x1, y1, z1, x2, y2, z2)
            self.prisms = [(0, 0, 0, W, H, D)]
            self.volumes = [] # List of (w, h, d) for placed boxes

        def _is_contained(self, p1, p2):
            # Check if p1 is fully contained in p2
            return (p1[0] >= p2[0] and p1[1] >= p2[1] and p1[2] >= p2[2] and
                    p1[3] <= p2[3] and p1[4] <= p2[4] and p1[5] <= p2[5])

        def _cleanup_prisms(self):
            # Remove prisms that are fully contained in another
            new_prisms = []
            for p in self.prisms:
                is_contained = False
                for q in self.prisms:
                    if p is q:
                        continue
                    if self._is_contained(p, q):
                        is_contained = True
                        break
                if not is_contained:
                    new_prisms.append(p)
            self.prisms = new_prisms

        def add_box(self, w, h, d):
            # Find best prism to place the box
            best_prism_idx = -1
            best_cost = float('inf')
            
            for i, (x1, y1, z1, x2, y2, z2) in enumerate(self.prisms):
                if (x2 - x1) >= w and (y2 - y1) >= h and (z2 - z1) >= d:
                    # Candidate position is bottom-left-front of this prism
                    # Cost: prefer lower Z, then Y, then X
                    cost = z1 * 10000 + y1 * 100 + x1
                    if cost < best_cost:
                        best_cost = cost
                        best_prism_idx = i

            if best_prism_idx == -1:
                return None

            x1, y1, z1, x2, y2, z2 = self.prisms[best_prism_idx]
            pos = (x1, y1, z1)
            
            # Split the prism
            bx1, by1, bz1 = x1, y1, z1
            bx2, by2, bz2 = x1 + w, y1 + h, z1 + d
            
            new_prisms = []
            
            if bx1 > x1:
                new_prisms.append((x1, y1, z1, bx1, y2, z2))
            if bx2 < x2:
                new_prisms.append((bx2, y1, z1, x2, y2, z2))
            if by1 > y1:
                new_prisms.append((bx1, y1, z1, bx2, by1, z2))
            if by2 < y2:
                new_prisms.append((bx1, by2, z1, bx2, y2, z2))
            if bz1 > z1:
                new_prisms.append((bx1, by1, z1, bx2, by2, bz1))
            if bz2 < z2:
                new_prisms.append((bx1, by1, bz2, bx2, by2, z2))

            del self.prisms[best_prism_idx]
            self.prisms.extend(new_prisms)
            self._cleanup_prisms()
            
            self.volumes.append((w, h, d))
            return pos

    def greedy_place(ordering, bin_selection='fullest'):
        placements = [None] * n
        bins = []
        bin_volumes = []

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                if bin_selection == 'fullest':
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
                else:
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi])
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = bins[bin_i].add_box(w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                new_bin = Bin(W, H, D)
                pos = new_bin.add_box(w, h, d)
                if pos is not None:
                    bins.append(new_bin)
                    bin_volumes.append(w * h * d)
                    placements[box_idx] = (new_bin_idx, pos[0], pos[1], pos[2], w, h, d)
                else:
                    bins.append(new_bin)
                    bin_volumes.append(w * h * d)
                    placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_improve(placements):
        # Skip local improvement to save time for more restarts
        return placements

    best_result = None
    best_score = float('inf')

    # Try different strategies
    strategies = [
        ('greedy_fullest', lambda order: greedy_place(order, 'fullest')),
        ('greedy_balance', lambda order: greedy_place(order, 'balance')),
    ]

    for strategy_name, strategy_func in strategies:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = strategy_func(order)

                if not validate_placement(placements):
                    continue

                score = compute_score(placements)

                if score < best_score:
                    best_score = score
                    best_result = placements[:]
            except Exception:
                continue

    if best_result is not None:
        return best_result

    # Fallback: one box per bin
    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
