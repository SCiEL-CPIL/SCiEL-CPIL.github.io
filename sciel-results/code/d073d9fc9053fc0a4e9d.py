        valid = True
        # Check x
        if x in current_products:
            valid = False
        else:
            for s in current_suffixes:
                p = x * s
                if p in current_products:
                    valid = False
                    break
        
        if valid:
             # Construct new_prods for updating the set
             new_prods = [x]
             for s in current_suffixes:
                 new_prods.append(x * s)
             # ... rest of include logic
