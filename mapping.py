def pixel_to_world(cx, cy, width, height, scale=0.05):
    world_x = (cx - width / 2) * scale
    world_y = (height - cy) * scale
    return world_x, world_y