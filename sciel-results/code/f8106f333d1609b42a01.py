            vi = _UNIT_VERTS + centers[i]
            dists = np.min(np.cross(oe, vi[None, :] - ov[:, None, :]), axis=1)
