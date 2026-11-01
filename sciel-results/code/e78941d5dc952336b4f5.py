            candidates = []
            temp_mask = not_in_set_mask
            while temp_mask:
                # Find lowest set bit
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                temp_mask ^= lsb
                if len(candidates) > 1000: # Limit list size for speed
                    break
