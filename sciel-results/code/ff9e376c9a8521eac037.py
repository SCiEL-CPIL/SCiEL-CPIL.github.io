    def check_subset_regular(adj_mask, mask):
        m = mask
        if m == 0: return False
        v0 = (m & -m).bit_length() - 1
        m ^= (1 << v0)
        d0 = (adj_mask[v0] & mask).bit_count()
        
        while m:
            v = (m & -m).bit_length() - 1
            m ^= (1 << v)
            d = (adj_mask[v] & mask).bit_count()
            if d != d0:
                return False
        return True
