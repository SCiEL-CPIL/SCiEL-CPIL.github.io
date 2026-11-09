<<<<<<< SEARCH
    def check_ap(colors, a, d):
        """Check if AP starting at a with difference d has >= 3 colors."""
        # a, a+d, a+2d, a+3d (1-indexed)
        # indices: a-1, a-1+d, a-1+2d, a-1+3d
        c = [colors[a-1], colors[a-1+d], colors[a-1+2d], colors[a-1+3d]]
        return len(set(c)) >= 3
=======
    def check_ap(colors, a, d):
        """Check if AP starting at a with difference d has >= 3 colors."""
        # a, a+d, a+2d, a+3d (1-indexed)
        # indices: a-1, a-1+d, a-1+2d, a-1+3d
        c = [colors[a-1], colors[a-1+d], colors[a-1+2*d], colors[a-1+3*d]]
        return len(set(c)) >= 3
>>>>>>> REPLACE
