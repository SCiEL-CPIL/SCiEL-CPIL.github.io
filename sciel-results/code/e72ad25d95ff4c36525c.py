    def update_cover(p, old_r, new_r):
        # Use numpy slicing
        old_slice = slice(old_r, N+1, p)
        new_slice = slice(new_r, N+1, p)
        
        cover_count[old_slice] -= 1
        cover_count[new_slice] += 1
        
        # Update removed_mask
        # removed_mask[x] is True if cover_count[x] > 0
        # We can just recompute removed_mask = cover_count > 0
        # This is O(N).
        # Is it faster than updating individually?
        # N=5000. O(N) is fast in Numpy.
        removed_mask[:] = cover_count > 0
