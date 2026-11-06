    def ap_is_bad(colors, ap_idx):
        c0 = colors[ap_idx[0]]
        c1 = colors[ap_idx[1]]
        c2 = colors[ap_idx[2]]
        c3 = colors[ap_idx[3]]
        
        if c0 == c1:
            if c2 == c0:
                return True # 1 or 2 colors
            else:
                # c0, c2 distinct
                return c3 == c0 or c3 == c2
        else:
            # c0, c1 distinct
            if c2 == c0:
                return c3 == c0 or c3 == c1
            elif c2 == c1:
                return c3 == c0 or c3 == c1
            else:
                return False
    