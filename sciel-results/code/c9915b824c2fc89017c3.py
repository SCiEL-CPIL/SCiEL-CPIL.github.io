<<<<<<< SEARCH
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
=======
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 12.0: # (2*sqrt(3))^2 * 1.5 ? No, max distance between centers for contact is 2*sqrt(3) ~ 3.46. Squared is 12.
            return True
>>>>>>> REPLACE
