import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

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

    class MEP:
        """Maximal Empty Prism: defined by (x, y, z) origin and (w, h, d) size."""
        __slots__ = ['x', 'y', 'z', 'w', 'h', 'd']
        
        def __init__(self, x, y, z, w, h, d):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            self.h = h
            self.d = d
        
        def volume(self):
            return self.w * self.h * self.d
        
        def fits(self, bw, bh, bd):
            return self.w >= bw and self.h >= bh and self.d >= bd
        
        def contains(self, x, y, z, w, h, d):
            return (self.x <= x and self.y <= y and self.z <= z and
                    x + w <= self.x + self.w and
                    y + h <= self.y + self.h and
                    z + d <= self.z + self.d)
    
    class Bin3D_MEP:
        """A bin using Maximal Empty Prisms for space management."""
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            self.meps = [MEP(0, 0, 0, W, H, D)]
            self.boxes = []
            self.volume = 0
        
        def _split_mep(self, mep, x, y, z, bw, bh, bd):
            """Split a MEP around a placed box, returning new MEPS."""
            new_meps = []
            
            # Left side
            if mep.x < x:
                new_meps.append(MEP(mep.x, mep.y, mep.z, x - mep.x, mep.h, mep.d))
            
            # Right side
            if x + bw < mep.x + mep.w:
                new_meps.append(MEP(x + bw, mep.y, mep.z, mep.x + mep.w - x - bw, mep.h, mep.d))
            
            # Front side
            if mep.y < y:
                new_meps.append(MEP(mep.x, mep.y, mep.z, mep.w, y - mep.y, mep.d))
            
            # Back side
            if y + bh < mep.y + mep.h:
                new_meps.append(MEP(mep.x, y + bh, mep.z, mep.w, mep.y + mep.h - y - bh, mep.d))
            
            # Bottom side
            if mep.z < z:
                new_meps.append(MEP(mep.x, mep.y, mep.z, mep.w, mep.h, z - mep.z))
            
            # Top side
            if z + bd < mep.z + mep.d:
                new_meps.append(MEP(mep.x, mep.y, z + bd, mep.w, mep.h, mep.z + mep.d - z - bd))
            
            return new_meps
        
        def place_box(self, x, y, z, bw, bh, bd):
            """Place a box and update MEPS."""
            # Find the MEP that contains this position
            containing_mep = None
            for mep in self.meps:
                if mep.contains(x, y, z, bw, bh, bd):
                    containing_mep = mep
                    break
            
            if containing_mep is None:
                return False
            
            # Remove the containing MEP and add split MEPS
            self.meps.remove(containing_mep)
            new_meps = self._split_mep(containing_mep, x, y, z, bw, bh, bd)
            
            # Remove any new MEPs that are contained in other new MEPs
            filtered = []
            for mep in new_meps:
                if mep.volume() <= 0:
                    continue
                contained = False
                for other in filtered:
                    if other.contains(mep.x, mep.y, mep.z, mep.w, mep.h, mep.d):
                        contained = True
                        break
                if not contained:
                    filtered.append(mep)
            
            self.meps.extend(filtered)
            self.boxes.append((x, y, z, bw, bh, bd))
            self.volume += bw * bh * bd
            return True
        
        def find_best_position(self, bw, bh, bd, strategy='best_fit'):
            """Find the best position for a box using MEPs."""
            if bw > self.W or bh > self.H or bd > self.D:
                return None
            
            best_pos = None
            best_score = float('inf')
            
            for mep in self.meps:
                if not mep.fits(bw, bh, bd):
                    continue
                
                # Try placing at the origin of this MEP
                x, y, z = mep.x, mep.y, mep.z
                
                # Calculate remaining volume after placement
                remaining_vol = mep.volume() - bw * bh * bd
                
                if strategy == 'best_fit':
                    # Minimize remaining volume (tightest fit)
                    score = remaining_vol
                elif strategy == 'bottom_left':
                    # Prefer positions close to bottom-left-front
                    score = z * (self.H + 1) * (self.W + 1) + y * (self.W + 1) + x
                elif strategy == 'combined':
                    # Combine bottom-left preference with tight fit
                    base = z * 1000 + y * 10 + x
                    score = base + remaining_vol * 0.1
                else:
                    score = remaining_vol
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
            
            return best_pos

    def greedy_place_mep(ordering, strategy):
        """Greedy placement using MEPs."""
        placements = [None] * n
        bins = []

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                # Try bins with most volume used first
                bin_indices = sorted(range(len(bins)), key=lambda bi: bins[bi].volume, reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = bins[bin_i].find_best_position(w, h, d, strategy)
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].place_box(x, y, z, w, h, d)
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                new_bin = Bin3D_MEP(W, H, D)
                new_bin.place_box(0, 0, 0, w, h, d)
                bins.append(new_bin)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    best_result = None
    best_score = float('inf')

    strategies = ['best_fit', 'bottom_left', 'combined']

    for strategy in strategies:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = greedy_place_mep(order, strategy)

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
