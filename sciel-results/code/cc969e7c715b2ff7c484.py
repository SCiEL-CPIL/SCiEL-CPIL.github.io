        if idx in prod_set:
            collision = True
        else:
            new_suffixes = [idx]
            for p in suffix_list:
                val = p * idx
                new_suffixes.append(val)
                if val in prod_set:
                    collision = True
                    break # No need to check further or build list
            # If we broke, new_suffixes is incomplete, but we won't use it.
            # If we didn't break, new_suffixes is complete.
