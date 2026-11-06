    def is_non_colorable(active_mask):
        for bm in bad_masks:
            if (bm & active_mask) == 0:
                return False
        return True
    