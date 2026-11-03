<<<<<<< SEARCH
    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
=======
    for vkey in queue:
        if time.time() > _deadline:
            break
        side, coords = vkey
        vidx = label[vkey]
>>>>>>> REPLACE
