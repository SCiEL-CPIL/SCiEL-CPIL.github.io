<<<<<<< SEARCH
    # Main packing: try to pack boxes in given order
    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = []
        result = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            if deadline_check():
                break
            pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result
=======
    # Main packing: try to pack boxes in given order
    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = []
        result = [None] * len(boxes)
        placed_indices = []  # Track which box index is at each position in placed
        
        for idx in order:
            w, h, d = boxes[idx]
            if deadline_check():
                break
            pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                placed_indices.append(idx)
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result, placed, placed_indices
>>>>>>> REPLACE

