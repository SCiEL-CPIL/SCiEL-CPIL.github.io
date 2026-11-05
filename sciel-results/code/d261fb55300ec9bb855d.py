            a_s = a & s_mask
            for j in range(i + 1, m):
                b = family_masks[j]
                if (a & b) == a_s and (b & s_mask) == a_s:
