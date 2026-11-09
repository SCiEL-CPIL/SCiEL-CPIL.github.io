<<<<<<< SEARCH
        def find_position(w, h, d):
            """Find a valid position for a box with dimensions w, h, d."""
            # Generate candidate positions
            candidates = [(0, 0, 0)]

            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Place to the right
                candidates.append((px + pw, py, pz))
                # Place in front
                candidates.append((px, py + ph, pz))
                # Place above
                candidates.append((px, py, pz + pd))
                # Combinations of two offsets
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
                # Also try positions aligned with existing box edges
                candidates.append((px, py, pz))
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                # Additional: try to fill gaps by aligning with other box edges
                for (px2, py2, pz2, pw2, ph2, pd2) in placed_boxes[:5]:  # Limit for speed
                    candidates.append((max(px, px2), py, pz))
                    candidates.append((px, max(py, py2), pz))
                    candidates.append((px, py, max(pz, pz2)))

            # Deduplicate and filter
            seen = set()
            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if (cx, cy, cz) in seen:
                    continue
                seen.add((cx, cy, cz))
                if can_place(cx, cy, cz, w, h, d):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            # Score: prefer positions that minimize the "bounding box" growth
            # and keep things compact. We use a weighted score.
            def pos_score(p):
                cx, cy, cz = p
                # Primary: minimize z (height), then y, then x
                # Secondary: prefer positions that are close to existing boxes (compactness)
                # Tertiary: prefer positions that leave more space in the remaining dimensions
                primary = (cz, cy, cx)
                # Remaining space after placing
                rem_x = W - (cx + w)
                rem_y = H - (cy + h)
                rem_z = D - (cz + d)
                # We want to maximize remaining space in the largest dimension
                max_rem = max(rem_x, rem_y, rem_z)
                # Penalize if we're creating a large void
                return (primary, -max_rem)

            valid_candidates.sort(key=pos_score)
            return valid_candidates[0]
=======
        def find_position(w, h, d):
            """Find a valid position for a box with dimensions w, h, d."""
            # Generate candidate positions using extreme-point approach
            candidates = set()
            candidates.add((0, 0, 0))

            # For each placed box, generate extreme points (corners that could be placement sites)
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # 8 corners of each placed box - these are potential placement positions
                for cx in [px, px + pw]:
                    for cy in [py, py + ph]:
                        for cz in [pz, pz + pd]:
                            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                                candidates.add((cx, cy, cz))
                
                # Also add face-aligned positions (slightly offset from faces)
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                
                # Try to fill gaps: align with edges of other placed boxes
                for (px2, py2, pz2, pw2, ph2, pd2) in placed_boxes:
                    if (px2, py2, pz2) == (px, py, pz):
                        continue
                    # Align x with another box's edge
                    if px2 < px + pw and px2 + pw2 > px:
                        candidates.add((px2, py, pz))
                        candidates.add((px2 + pw2, py, pz))
                    # Align y with another box's edge  
                    if py2 < py + ph and py2 + ph2 > py:
                        candidates.add((px, py2, pz))
                        candidates.add((px, py2 + ph2, pz))
                    # Align z with another box's edge
                    if pz2 < pz + pd and pz2 + pd2 > pz:
                        candidates.add((px, py, pz2))
                        candidates.add((px, py, pz2 + pd2))

            # Deduplicate and filter
            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if can_place(cx, cy, cz, w, h, d):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            # Score: prefer positions that minimize "bounding box" growth and keep things compact
            def pos_score(p):
                cx, cy, cz = p
                # Primary: minimize z (height), then y, then x
                primary = (cz, cy, cx)
                # Secondary: compactness - prefer positions close to existing boxes
                if placed_boxes:
                    min_dist = min(
                        abs(cx - px) + abs(cy - py) + abs(cz - pz)
                        for (px, py, pz, pw, ph, pd) in placed_boxes
                    )
                else:
                    min_dist = 0
                # Tertiary: prefer positions that leave more space in the remaining dimensions
                rem_x = W - (cx + w)
                rem_y = H - (cy + h)
                rem_z = D - (cz + d)
                max_rem = max(rem_x, rem_y, rem_z)
                return (primary, min_dist, -max_rem)

            valid_candidates.sort(key=pos_score)
            return valid_candidates[0]
>>>>>>> REPLACE

