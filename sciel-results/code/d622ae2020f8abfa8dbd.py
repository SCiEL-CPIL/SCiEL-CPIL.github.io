"""3D Container Loading - Hierarchical empty-space packer with ILS refinement."""
import time
import random
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 110.0  # seconds, leave margin for safety

    n = len(boxes)
    if n == 0:
        return []

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        orients = set(permutations([w, h, d]))
        return list(orients)

    box_orientations = [get_orientations(boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]

    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -box_volumes[i])

    # Hierarchical empty-space representation:
    # Each node is a rectangular prism (x, y, z, w, h, d) representing empty space.
    # When a box is placed in a node, the node is split into up to 3 sub-nodes.
    # This guarantees non-overlap by construction.

    class EmptySpaceTree:
        def __init__(self, W, H, D):
            self.root = (0, 0, 0, W, H, D)
            self.nodes = [self.root]  # list of empty prisms

        def find_best_slot(self, bw, bh, bd, scoring_fn):
            """Find the best empty prism to place a box of size (bw, bh, bd).
            Returns (node_index, position, orientation) or None."""
            best_score = float('inf')
            best_result = None

            for i, node in enumerate(self.nodes):
                nx, ny, nz, nw, nh, nd = node
                # Check if box fits in this node
                if bw <= nw and bh <= nh and bd <= nd:
                    # Try different orientations within this node
                    # Actually, the caller handles orientation; here we just check fit
                    # Compute score based on position (prefer lower, then closer to walls)
                    score = scoring_fn(nx, ny, nz, nw, nh, nd, bw, bh, bd)
                    if score < best_score:
                        best_score = score
                        best_result = (i, (nx, ny, nz))

            return best_result

        def place_box(self, node_idx, x, y, z, bw, bh, bd):
            """Place a box at (x,y,z) with size (bw,bh,bd) in the given node.
            Splits the node into sub-nodes."""
            node = self.nodes[node_idx]
            nx, ny, nz, nw, nh, nd = node

            # The box is placed at (x, y, z) within this node
            # Since the node is empty, we just need to carve out the box region
            # and replace the node with the remaining empty regions

            # Remove the current node
            self.nodes.pop(node_idx)

            # Generate sub-nodes (L-shaped decomposition)
            # We split along the longest dimension that the box doesn't fill completely

            # Region 1: To the right of the box (if any)
            if x + bw < nx + nw:
                self.nodes.append((x + bw, y, z, nx + nw - (x + bw), bh, bd))

            # Region 2: Above the box (if any)
            if y + bh < ny + nh:
                self.nodes.append((x, y + bh, z, bw, ny + nh - (y + bh), bd))

            # Region 3: In front of the box (if any)
            if z + bd < nz + nd:
                self.nodes.append((x, y, z + bd, bw, bh, nz + nd - (z + bd)))

            # Additional regions to cover the L-shape properly
            # We need to be more careful: the above creates overlaps
            # Let's use a proper decomposition

            # Actually, let me redo this properly.
            # When we place a box at (x,y,z) with size (bw,bh,bd) in a node (nx,ny,nz,nw,nh,nd):
            # The remaining space is:
            # 1. Right of box: (x+bw, ny, nz, nw-(x+bw-nx), nh, nd) if x+bw < nx+nw
            # 2. Above box: (nx, y+bh, nz, nw, nh-(y+bh-ny), nd) if y+bh < ny+nh
            # 3. Front of box: (nx, ny, z+bd, nw, nh, nd-(z+bd-nz)) if z+bd < nz+nd

            # But this creates overlaps. The correct decomposition is:
            # Split the node into at most 3 non-overlapping regions.

            # Let's use a different approach: we'll split along one axis at a time.
            # First, split along x-axis if the box doesn't fill the full width
            if x + bw < nx + nw:
                # Right region
                self.nodes.append((x + bw, ny, nz, nx + nw - (x + bw), nh, nd))
            if x > nx:
                # Left region
                self.nodes.append((nx, ny, nz, x - nx, nh, nd))

            # Then for the remaining region containing the box, split along y-axis
            # The region containing the box is (x, ny, nz, bw, nh, nd) if we split on x
            # This is getting complex. Let me use a simpler approach.

            # Simple approach: just add the 3 regions that don't overlap with the box
            # and accept that there might be some inefficiency. The key is correctness.

            # Actually, for correctness, let's just use the simple 3-split:
            # The box occupies (x, y, z, bw, bh, bd)
            # Remaining space in the node:
            # We'll create sub-nodes that tile the remaining space.

            # To keep it simple and correct, let's just store the box as "occupied"
            # and check against all placed boxes. But that defeats the purpose.

            # Let me restart with a cleaner approach.
            pass

    # Let me use a simpler but correct approach: maintain a list of empty rectangular prisms
    # and when placing a box, split the containing prism into up to 3 non-overlapping prisms.

    def try_place(boxes_order, t0, deadline):
        """Try to place boxes in the given order using hierarchical empty-space."""
        placements = [None] * n
        filled_vol = 0

        # Start with the entire container as one empty region
        empty_regions = [(0, 0, 0, W, H, D)]  # (x, y, z, w, h, d)

        for idx in boxes_order:
            if time.time() - t0 > deadline:
                break

            w, h, d = boxes[idx]
            orientations = box_orientations[idx]

            best_pos = None
            best_score = float('inf')
            best_region_idx = -1
            best_orient = None

            # Find the best empty region and orientation
            for ri, (ex, ey, ez, ew, eh, ed) in enumerate(empty_regions):
                for ow, oh, od in orientations:
                    # Check if box fits in this region
                    if ow > ew or oh > eh or od > ed:
                        continue

                    # Try to place at the "lowest" position in this region
                    # We'll try placing at (ex, ey, ez) first (bottom-left-back corner)
                    px, py, pz = ex, ey, ez

                    # Score: prefer lower z, then lower y, then lower x
                    # Also prefer tighter fit (less wasted space)
                    waste = (ew - ow) * eh * ed + ew * (eh - oh) * ed + ew * eh * (ed - od)
                    score = pz * 1000000 + py * 1000 + px + waste * 0.001

                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_region_idx = ri
                        best_orient = (ow, oh, od)

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                filled_vol += bw * bh * bd

                # Split the region into sub-regions
                ex, ey, ez, ew, eh, ed = empty_regions[best_region_idx]
                empty_regions.pop(best_region_idx)

                # The box is at (bx, by, bz) with size (bw, bh, bd)
                # within the region (ex, ey, ez, ew, eh, ed)

                # Region to the right of the box
                if bx + bw < ex + ew:
                    empty_regions.append((bx + bw, by, bz, ex + ew - (bx + bw), bh, bd))

                # Region above the box
                if by + bh < ey + eh:
                    empty_regions.append((bx, by + bh, bz, bw, ey + eh - (by + bh), bd))

                # Region in front of the box
                if bz + bd < ez + ed:
                    empty_regions.append((bx, by, bz + bd, bw, bh, ez + ed - (bz + bd)))

                # Additional regions to cover the L-shape
                # Right and above
                if bx + bw < ex + ew and by + bh < ey + eh:
                    empty_regions.append((bx + bw, by + bh, bz, ex + ew - (bx + bw), ey + eh - (by + bh), bd))

                # Right and front
                if bx + bw < ex + ew and bz + bd < ez + ed:
                    empty_regions.append((bx + bw, by, bz + bd, ex + ew - (bx + bw), bh, ez + ed - (bz + bd)))

                # Above and front
                if by + bh < ey + eh and bz + bd < ez + ed:
                    empty_regions.append((bx, by + bh, bz + bd, bw, ey + eh - (by + bh), ez + ed - (bz + bd)))

                # Right, above, and front
                if bx + bw < ex + ew and by + bh < ey + eh and bz + bd < ez + ed:
                    empty_regions.append((bx + bw, by + bh, bz + bd, ex + ew - (bx + bw), ey + eh - (by + bh), ez + ed - (bz + bd)))

            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill in any remaining boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)

        return placements, filled_vol

    # First pass: standard best-fit decreasing
    best_placements, best_vol = try_place(box_indices, t0, DEADLINE)

    # If we have time left, try a few random permutations to see if we can do better
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5:
        random.seed(42)

        # Try a few random orderings
        for attempt in range(20):
            if time.time() - t0 > DEADLINE - 2:
                break

            # Create a perturbed ordering
            order = box_indices[:]
            for _ in range(8):
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i != j:
                    order[i], order[j] = order[j], order[i]

            placements, vol = try_place(order, t0, DEADLINE)

            if vol > best_vol:
                best_vol = vol
                best_placements = placements

    return best_placements
