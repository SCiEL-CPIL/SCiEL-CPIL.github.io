    def has_3term_ap(C, x):
        # Check if x is mean of 2 others
        for i in range(len(C)):
            for j in range(i+1, len(C)):
                if C[i] + C[j] == 2 * x:
                    return True
        # Check if a in C is mean of x and another in C
        for a in C:
            b = 2 * a - x
            if b in C and b != a:
                return True
        return False

    def check_validity(C, x):
        if has_3term_ap(C, x):
            return False
        if is_valid_subset_mean(C, x) and is_valid_existing_mean(C, x):
            return True
        return False
