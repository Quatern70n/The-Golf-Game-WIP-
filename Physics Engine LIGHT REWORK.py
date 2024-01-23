import pygame
from math import sqrt, pi, tau, e, sin, cos, tan, atan, atan2

all_vectors = []

# Special Functions

def clamp(low, n, high):

    return max(low, min(n, high))


def rgba_to_rgb(r, g, b, a):
    if a >= 0:
        return [(255 - r) * a + r, (255 - g) * a + g, (255 - b) * a + b]
    else:
        return [r * a + r, g * a + g, b * a + b]

#

# Vectors Library

def vector(x, y, z, position=[0, 0, 0], color=[255, 255, 255]):
    all_vectors.append({"x": x, "y": y, "z": z, "pos": position, "col": color})
    return {"x": x, "y": y, "z": z, "pos": position, "col": color}


def vector_recolor_by_scalar(v, i):
    v["col"] = [clamp(1, v["col"][0] * i, 255), clamp(1, v["col"][1] * i, 255), clamp(1, v["col"][2] * i, 255)]


def vector_recolor_by_scalar_rgba(v, i):
    v["col"] = rgba_to_rgb(round(v["col"][0]), round(v["col"][1]), round(v["col"][2]), i)


def position_scalar_product(pos, i):
    return [pos[0] * i, pos[1] * i, pos[2] * i]


def position_summation(pos1, pos2):
    return [pos1[0] + pos2[0], pos1[1] + pos2[1], pos1[2] + pos2[2]]


def vector_scalar_product(v, i):
    return vector(v["x"] * i, v["y"] * i, v["z"] * i, v["pos"], v["col"])


def vector_summation(v1, v2):
    return vector(v1["x"] + v2["x"], v1["y"] + v2["y"], v1["z"] + v2["z"], v1["pos"], list(arithmetic_color(v1, v2)))


def vector_subtract(v1, v2):
    return vector(v1["x"] - v2["x"], v1["y"] - v2["y"], v1["z"] - v2["z"], v1["pos"], list(arithmetic_color(v1, v2)))


def vectors_isequal(v1, v2):
    if v1["x"] == v2["x"] and \
            v1["y"] == v2["y"] and \
            v1["z"] == v2["z"]:
        return True
    else:
        return False


def vectors_distance(v1, v2):
    dx = v2["x"] - v1["x"]
    dy = v2["y"] - v1["y"]
    dz = v2["z"] - v1["z"]

    return sqrt(dx ** 2 + dy ** 2 + dz ** 2)

#

# Special point and polygons functions

def pair_minimum_distance(pair):
    pair_distance = [camera_point_info(pair[0])[0], camera_point_info(pair[1])[0]]
    return min(pair_distance)


def polygon_minimum_distance(polygon):
    polygon_distance = [camera_point_info(polygon[0])[0], camera_point_info(polygon[1])[0],
                        camera_point_info(polygon[2])[0]]
    return sum(polygon_distance) / 3


def polygon_minimum_distance_point(polygon, point):
    point1 = polygon[0]
    point2 = polygon[1]
    point3 = polygon[2]

    dx1, dy1, dz1 = point1["x"] - point["x"] + point1["pos"][0], point1["y"] - point["y"] + point1["pos"][1], point1[
        "z"] - point["z"] + point1["pos"][2]
    dx2, dy2, dz2 = point2["x"] - point["x"] + point2["pos"][0], point2["y"] - point["y"] + point2["pos"][1], point2[
        "z"] - point["z"] + point2["pos"][2]
    dx3, dy3, dz3 = point3["x"] - point["x"] + point3["pos"][0], point3["y"] - point["y"] + point3["pos"][1], point3[
        "z"] - point["z"] + point3["pos"][2]

    distance1 = sqrt(dx1 ** 2 + dy1 ** 2 + dz1 ** 2)
    distance2 = sqrt(dx2 ** 2 + dy2 ** 2 + dz2 ** 2)
    distance3 = sqrt(dx3 ** 2 + dy3 ** 2 + dz3 ** 2)

    distance = (distance1 + distance2 + distance3) / 3
    return distance

#

# Quaternions Library

def quaternion(w, x, y, z, position=[0, 0, 0], color=[255, 255, 255]):
    return {"w": w, "x": x, "y": y, "z": z, "pos": position, "col": color}


def quaternion_scalar_product(q, i):
    return quaternion(q["w"] * i, q["x"] * i, q["y"] * i, q["z"] * i, q["pos"], q["col"])


def quaternion_conjugate(q):
    return quaternion(q["w"], -q["x"], -q["y"], -q["z"], q["pos"], q["col"])


def quaternion_norm(q):
    return sqrt(q["w"] ** 2 + q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2)


def quaternion_normalize(q):
    return quaternion_scalar_product(q, 1 / quaternion_norm(q))


def quaternion_inverse(q):
    denominator = 1 / (quaternion_norm(q) ** 2)
    conjugate = quaternion_conjugate(q)
    return quaternion_scalar_product(conjugate, denominator)


def quaternion_vectorize(q, position, color):
    return vector(q["x"], q["y"], q["z"], position, color)


def quaternion_by_quaternion(q_a, q_b):
    w = q_a["w"] * q_b["w"] - q_a["x"] * q_b["x"] - q_a["y"] * q_b["y"] - q_a["z"] * q_b["z"]
    x = q_a["w"] * q_b["x"] + q_a["x"] * q_b["w"] + q_a["y"] * q_b["z"] - q_a["z"] * q_b["y"]
    y = q_a["w"] * q_b["y"] - q_a["x"] * q_b["z"] + q_a["y"] * q_b["w"] + q_a["z"] * q_b["x"]
    z = q_a["w"] * q_b["z"] + q_a["x"] * q_b["y"] - q_a["y"] * q_b["x"] + q_a["z"] * q_b["w"]
    return quaternion(w, x, y, z, q_a["pos"], q_a["col"])


def quaternion_by_vector(q, v):
    q_a, q_b = q, quaternion(w=0, x=v["x"], y=v["y"], z=v["z"])
    w = q_a["w"] * q_b["w"] - q_a["x"] * q_b["x"] - q_a["y"] * q_b["y"] - q_a["z"] * q_b["z"]
    x = q_a["w"] * q_b["x"] + q_a["x"] * q_b["w"] + q_a["y"] * q_b["z"] - q_a["z"] * q_b["y"]
    y = q_a["w"] * q_b["y"] - q_a["x"] * q_b["z"] + q_a["y"] * q_b["w"] + q_a["z"] * q_b["x"]
    z = q_a["w"] * q_b["z"] + q_a["x"] * q_b["y"] - q_a["y"] * q_b["x"] + q_a["z"] * q_b["w"]
    return quaternion(w, x, y, z, q["pos"], q["col"])


def quaternion_transform_vector(q, v):
    transform1 = quaternion_by_vector(q, v)
    transform2 = quaternion_by_quaternion(transform1, quaternion_inverse(q))
    return quaternion_vectorize(transform2, v["pos"], v["col"])


def versor(v, i, j, k):
    quaternion_x = quaternion(cos(i / 2), sin(i / 2), 0, 0)
    quaternion_y = quaternion(cos(j / 2), 0, sin(j / 2), 0)
    quaternion_z = quaternion(cos(k / 2), 0, 0, sin(k / 2))
    quaternion_xyz = quaternion_by_quaternion(quaternion_x, quaternion_by_quaternion(quaternion_y, quaternion_z))
    return quaternion_transform_vector(quaternion_xyz, v)


def massive_vector_transform(i, j, k, *args):
    transformed_vectors = []
    for vector in args:
        transformed_vectors.append(versor(vector, i, j, k))
    return transformed_vectors

#

# Matrix Library

def matrix(a, b, c, d, e, f, g, h, i):
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g, 'h': h, 'i': i}


def vector_by_matrix(v, m):
    x = v["x"] * m["a"] + v["y"] * m["b"] + v["z"] * m["c"]
    y = v["x"] * m["d"] + v["y"] * m["e"] + v["z"] * m["f"]
    z = v["x"] * m["g"] + v["y"] * m["h"] + v["z"] * m["i"]
    return vector(x, y, z, v["pos"], v["col"])


def rotor(v, x, y, z):
    matrix_xyz = matrix(cos(y) * cos(z), -sin(z) * cos(y), sin(y),
                        sin(x) * sin(y) * cos(z) + sin(z) * cos(x), -sin(x) * sin(y) * sin(z) + cos(x) * cos(z),
                        -sin(x) * cos(y),
                        sin(x) * sin(z) - sin(x) * cos(x) * cos(z), sin(x) * cos(z) + sin(y) * sin(z) * cos(x),
                        cos(x) * cos(x))

    return vector_by_matrix(v, matrix_xyz)

#

# Physics Library

g = 9.81 # m/s2
tick = 200 # t/s
tick_incriminate = 0

euler_force_constant = 1
euler_force_vector = vector(0, 0, 0)


def gravitation_transform(vectors, mass=0, reverse=False, repulse=False):
    transformed_vectors = []
    if reverse:
        constant = -g / tick ** 2
    else:
        constant = g / tick ** 2

    for vector_def in vectors:
        new_z = vector_def["z"] - constant
        new_pos_z = vector_def["pos"][2] - constant
        new_vector = vector(vector_def["x"], vector_def["y"], new_z,
                            [vector_def["pos"][0], vector_def["pos"][1], new_pos_z], vector_def["col"])

        transformed_vectors.append(new_vector)

    global tick_incriminate

    tick_incriminate += 1

    if repulse:
        tick_incriminate = 0

    pulse = vector(0, -g * mass * tick_incriminate, 0)

    return transformed_vectors, pulse


def force_transform(vectors, drag=0, pulse=vector(0, 0, 0), mass=0, reverse=False, repulse=False):
    transformed_vectors = []

    global euler_force_constant, euler_force_vector

    if repulse:
        euler_force_constant = 1
        euler_force_vector = vector_scalar_product(pulse, 1 / mass)

    euler_force_constant *= (1 - drag)

    if reverse:
        euler_force_vector_transformed = vector_scalar_product(euler_force_vector, -euler_force_constant)
    else:
        euler_force_vector_transformed = vector_scalar_product(euler_force_vector, euler_force_constant)

    for vector_def in vectors:
        new_x = vector_def["x"] - euler_force_vector_transformed["x"]
        new_y = vector_def["y"] - euler_force_vector_transformed["y"]
        new_z = vector_def["z"] - euler_force_vector_transformed["z"]
        new_pos_x = vector_def["pos"][0] - euler_force_vector_transformed["x"]
        new_pos_y = vector_def["pos"][1] - euler_force_vector_transformed["y"]
        new_pos_z = vector_def["pos"][2] - euler_force_vector_transformed["z"]

        new_vector = vector(new_x, new_y, new_z, [new_pos_x, new_pos_y, new_pos_z], vector_def["col"])

        transformed_vectors.append(new_vector)

    return transformed_vectors, euler_force_vector_transformed


def collision(vectors_1, vectors_2, velocity_1=vector(0, 0, 0), is_wedge=False):
    # Creating Collision Of The First Figure

    centre_of_mass_1_srt_coordinates_sum = [0, 0, 0]
    centre_of_mass_1_end_coordinates_sum = [0, 0, 0]

    for vector_def in vectors_1:
        centre_of_mass_1_srt_coordinates_sum[0] += vector_def["pos"][0]
        centre_of_mass_1_srt_coordinates_sum[1] += vector_def["pos"][1]
        centre_of_mass_1_srt_coordinates_sum[2] += vector_def["pos"][2]

        centre_of_mass_1_end_coordinates_sum[0] += vector_def["x"]
        centre_of_mass_1_end_coordinates_sum[1] += vector_def["y"]
        centre_of_mass_1_end_coordinates_sum[2] += vector_def["z"]

    centre_of_mass_1_srt_coordinates = [centre_of_mass_1_srt_coordinates_sum[0] // len(vectors_1),
                                        centre_of_mass_1_srt_coordinates_sum[1] // len(vectors_1),
                                        centre_of_mass_1_srt_coordinates_sum[2] // len(vectors_1)]

    centre_of_mass_1_end_coordinates = [centre_of_mass_1_end_coordinates_sum[0] // len(vectors_1),
                                        centre_of_mass_1_end_coordinates_sum[1] // len(vectors_1),
                                        centre_of_mass_1_end_coordinates_sum[2] // len(vectors_1)]

    centre_of_mass_1 = vector(centre_of_mass_1_end_coordinates[0],
                              centre_of_mass_1_end_coordinates[1],
                              centre_of_mass_1_end_coordinates[2],
                              centre_of_mass_1_srt_coordinates,
                              [255, 0, 255])

    distances_1 = []

    for vector_def in vectors_1:
        distances_1.append(vectors_distance(centre_of_mass_1, vector_def))

    center_distance_1 = sum(distances_1) / len(distances_1)

    cuboid_collision_1 = []

    for height in range(-1, 1, 2):
        for horizontal in range(-90, 180, 90):
            radian_angle = horizontal / 180 * pi
            x = center_distance_1 * cos(radian_angle) - center_distance_1 * sin(radian_angle)
            y = center_distance_1 * sin(radian_angle) + center_distance_1 * cos(radian_angle)
            z = center_distance_1 * height
            position = [centre_of_mass_1["x"], centre_of_mass_1["y"], centre_of_mass_1["z"]]

            new_vector_1 = vector(x, y, z, position, [0, 255, 0])

            cuboid_collision_1.append(new_vector_1)

    #

    # Checking Collision (the second figure is simple)\

    #   8 ######## 5         #                        #         z
    #   # #        # #       #   6 ######## 5         #         ^
    #   #   #      #   #     #   ##         ##        #      \  |
    #   #     #    #     #   #   # #        # #       #       \ |
    #   #       7 ######## 6 #   #  #       #  #      #        \|
    #   4 ######## 1       # #   4 ######## 1   #     # ------- 0 -------> x
    #     #     #    #     # #     #  #       #  #    #         |\
    #       #   #      #   # #       # #        # #   #         | \
    #         # #        # # #         ##         ##  #         |  \
    #           3 ######## 2 #           3 ######## 2 #             y

    if is_wedge:
        wedge_coefficient = (vectors_2[4]["z"] - vectors_2[4]["pos"][2]) - (vectors_2[0]["z"] - vectors_2[0]["pos"][2]) / \
                            (vectors_2[1]["y"] - vectors_2[1]["pos"][1]) - (vectors_2[0]["y"] - vectors_2[0]["pos"][1])
        for width in range(vectors_2[4]["x"] - vectors_2[4]["pos"][0], vectors_2[7]["x"] - vectors_2[7]["pos"][0]):
            for depth in range(vectors_2[4]["y"] - vectors_2[4]["pos"][1], vectors_2[5]["y"] - vectors_2[5]["pos"][1]):
                height = depth * wedge_coefficient
                for point in cuboid_collision_1:
                    if point["x"] == width and point["y"] == depth and point["z"] == height:
                        if vectors_2[4]["z"] - vectors_2[4]["pos"][2] > centre_of_mass_1["z"] > vectors_2[1]["z"] - vectors_2[1]["pos"][2]:
                            if vectors_2[1]["y"] - vectors_2[1]["pos"][0] > centre_of_mass_1["y"] > vectors_2[0]["y"] - vectors_2[0]["pos"][0]:
                                velocity_1["x"] = -velocity_1["x"]
                            elif vectors_2[5]["x"] - vectors_2[5]["pos"][0] > centre_of_mass_1["x"] > vectors_2[4]["x"] - vectors_2[4]["pos"][0] and centre_of_mass_1["y"] > vectors_2[0]["y"] - vectors_2[0]["pos"][0]:
                                velocity_1["y"] = -velocity_1["y"]
                            elif vectors_2[5]["x"] - vectors_2[5]["pos"][0] > centre_of_mass_1["x"] > vectors_2[4]["x"] - vectors_2[4]["pos"][0] and centre_of_mass_1["y"] < vectors_2[0]["y"] - vectors_2[0]["pos"][0]:
                                angle = atan(wedge_coefficient) * 2
                                velocity_1["z"] = velocity_1["z"] * cos(angle) - sqrt(velocity_1["x"] ** 2 + velocity_1["y"] ** 2) * sin(angle)
    else:
        for width in range(vectors_2[4]["x"] - vectors_2[4]["pos"][0], vectors_2[7]["x"] - vectors_2[7]["pos"][0]):
            for depth in range(vectors_2[4]["y"] - vectors_2[4]["pos"][1], vectors_2[5]["y"] - vectors_2[5]["pos"][1]):
                for height in range(vectors_2[4]["z"] - vectors_2[4]["pos"][2], vectors_2[1]["z"] - vectors_2[1]["pos"][2]):
                    for point in cuboid_collision_1:
                        if point["x"] == width and point["y"] == depth and point["z"] == height:
                            if centre_of_mass_1 > vectors_2[4]["z"] - vectors_2[4]["pos"][2] or centre_of_mass_1["z"] < vectors_2[1]["z"] - vectors_2[1]["pos"][2]:
                                velocity_1["z"] = -velocity_1["z"]
                            elif vectors_2[4]["z"] - vectors_2[4]["pos"][2] > centre_of_mass_1["z"] > vectors_2[1]["z"] - vectors_2[1]["pos"][2]:
                                if vectors_2[4]["x"] - vectors_2[4]["pos"][0] > centre_of_mass_1["x"] > vectors_2[7]["x"] - vectors_2[7]["pos"][0]:
                                    velocity_1["y"] = -velocity_1["y"]
                                elif vectors_2[4]["y"] - vectors_2[4]["pos"][0] > centre_of_mass_1["y"] > vectors_2[5]["y"] - vectors_2[5]["pos"][0]:
                                    velocity_1["x"] = -velocity_1["x"]

    return velocity_1
    #

#

# Special Camera Functions and Statements

camera_position = vector(0, 0, 0)
camera_euler_angles = {"precession": 0.0, "nutation": 0.0, "natural": 0.0}
lights = []


def camera_reposition(x=0, y=0, z=0):
    camera_position["x"] = x
    camera_position["y"] = y
    camera_position["z"] = z


def camera_rotation(p=0.0, n=0.0, s=0.0):
    camera_euler_angles["precession"] = p
    camera_euler_angles["nutation"] = n
    camera_euler_angles["natural"] = s

    for index, vector_def in enumerate(all_vectors):
        local_x = vector_def["x"] - vector_def["pos"][0] - camera_position["x"]
        local_y = vector_def["y"] - vector_def["pos"][1] - camera_position["y"]
        local_z = vector_def["z"] - vector_def["pos"][2] - camera_position["z"]
        new_vector = vector(local_x, local_y, local_z,
                            [-camera_position["x"], -camera_position["y"], -camera_position["z"]], vector_def["col"])
        new_vector_transform = rotor(new_vector, camera_euler_angles["precession"], camera_euler_angles["nutation"],
                                     camera_euler_angles["natural"])
        new_vector_return = vector(new_vector_transform["x"] + vector_def["pos"][0] + camera_position["x"],
                                   new_vector_transform["y"] + vector_def["pos"][1] + camera_position["y"],
                                   new_vector_transform["z"] + vector_def["pos"][2] + camera_position["z"],
                                   vector_def["pos"], vector_def["col"])

        all_vectors[index] = new_vector_return


def camera_point_info(point):
    dx = point["x"] - point["pos"][0] - camera_position["x"]
    dy = point["y"] - point["pos"][1] - camera_position["y"]
    dz = point["z"] - point["pos"][2] + camera_position["z"]
    distance_2d = sqrt(dx ** 2 + dy ** 2)
    distance_3d = sqrt(distance_2d ** 2 + dz ** 2)
    angle_horizontal = atan2(dx, dy)
    angle_vertical = atan2(dz, distance_2d)
    return distance_3d, angle_horizontal, angle_vertical, distance_2d


def camera_point_info_perspective(point):
    dx = point["x"] - point["pos"][0] + camera_position["x"]
    dy = point["y"] - point["pos"][1] - camera_position["y"]
    dz = point["z"] - point["pos"][2] + camera_position["z"]
    distance_2d = sqrt(dx ** 2 + dy ** 2)
    distance_3d = sqrt(distance_2d ** 2 + dz ** 2)

    tan_xy = dx / dy
    tan_xyz = dz / distance_2d

    return distance_2d, distance_3d, tan_xy, tan_xyz

#

# Special Graphic Functions

def arithmetic_color(v1, v2):
    return (v1["col"][0] + v2["col"][0]) // 2, (v1["col"][1] + v2["col"][1]) // 2, (v1["col"][2] + v2["col"][2]) // 2


def arithmetic_color3(v1, v2, v3):
    return ((v1["col"][0] + v2["col"][0] + v3["col"][0]) // 3, (v1["col"][1] + v2["col"][1] + v3["col"][1]) // 3,
            (v1["col"][2] + v2["col"][2] + v3["col"][2]) // 3)


def vector_projection(v, screen, size):
    pygame.draw.line(screen, tuple(v["col"]),
                     (size[0] // 2 - camera_position["x"], size[1] // 2 - camera_position["z"]),
                     (size[0] // 2 + v["x"] - camera_position["x"] + v["pos"][0],
                      size[1] // 2 - v["z"] - camera_position["z"] + v["pos"][2]))


def vector_projection2(v1, v2, screen, size):
    pygame.draw.line(screen, arithmetic_color(v1, v2),
                     (size[0] // 2 + v1["x"] - camera_position["x"] + v1["pos"][0],
                      size[1] // 2 - v1["z"] - camera_position["z"] + v1["pos"][2]),
                     (size[0] // 2 + v2["x"] - camera_position["x"] + v2["pos"][0],
                      size[1] // 2 - v2["z"] - camera_position["z"] + v2["pos"][2]))


def polygon_projection(v1, v2, v3, screen, size, perspective=False, fov=45):
    tan2d_v1, tan3d_v1 = camera_point_info_perspective(v1)[2], camera_point_info_perspective(v1)[3]
    tan2d_v2, tan3d_v2 = camera_point_info_perspective(v2)[2], camera_point_info_perspective(v2)[3]
    tan2d_v3, tan3d_v3 = camera_point_info_perspective(v3)[2], camera_point_info_perspective(v3)[3]

    fov_constant = 1 / tan(fov / 180 * pi)

    perspective_h1, perspective_v1 = tan2d_v1 * size[0] / 2 * fov_constant, tan3d_v1 * size[1] / 2 * fov_constant
    perspective_h2, perspective_v2 = tan2d_v2 * size[0] / 2 * fov_constant, tan3d_v2 * size[1] / 2 * fov_constant
    perspective_h3, perspective_v3 = tan2d_v3 * size[0] / 2 * fov_constant, tan3d_v3 * size[1] / 2 * fov_constant

    if perspective:
        projection_h1, projection_v1 = perspective_h1, perspective_v1
        projection_h2, projection_v2 = perspective_h2, perspective_v2
        projection_h3, projection_v3 = perspective_h3, perspective_v3
    else:
        projection_h1, projection_v1 = v1["x"], v1["z"]
        projection_h2, projection_v2 = v2["x"], v2["z"]
        projection_h3, projection_v3 = v3["x"], v3["z"]

    point1 = (size[0] // 2 + projection_h1 - camera_position["x"] + v1["pos"][0])
    point2 = (size[1] // 2 - projection_v1 - camera_position["z"] + v1["pos"][2])
    point3 = (size[0] // 2 + projection_h2 - camera_position["x"] + v2["pos"][0])
    point4 = (size[1] // 2 - projection_v2 - camera_position["z"] + v2["pos"][2])
    point5 = (size[0] // 2 + projection_h3 - camera_position["x"] + v3["pos"][0])
    point6 = (size[1] // 2 - projection_v3 - camera_position["z"] + v3["pos"][2])

    pygame.draw.polygon(screen, arithmetic_color3(v1, v2, v3),
                        points=((point1, point2), (point3, point4), (point5, point6)))


def massive_vector_projection(screen, size, vectors):
    for vector in vectors:
        vector_projection(vector, screen, size)


def light_source(x=camera_position["x"], y=camera_position["y"], z=camera_position["z"], radius=100, diffusion=1.0):
    lights.append({"x": x, "y": y, "z": z, "rad": radius, "dif": diffusion})


def points_list_sequence(*points):
    points_dict = {}
    counter = 1
    for point in points:
        points_dict[f"{counter}"] = point
        counter += 1
    return points_dict


def point_render(screen, size, points_list, sequence, light=False, perspective=False):
    polygons_list = []
    light_values = []
    counter = 0
    state = True

    for index in sequence:
        polygons_list.append([points_list[str(index[0])], points_list[str(index[1])], points_list[str(index[2])]])

    polygons_list = sorted(polygons_list, key=lambda x: polygon_minimum_distance(x), reverse=True)

    for polygon in polygons_list:
        if light:
            if lights:
                for light1 in lights:
                    distance = polygon_minimum_distance_point(polygon, light1)
                    value = clamp(-1, (light1["rad"] - distance) / distance, 1) * light1["dif"]
                    if counter < len(lights) and state:
                        light_values.append(value)
                    elif counter == len(lights):
                        counter = 0
                        state = False
                    elif counter <= len(lights) and not state:
                        light_values[counter - 1] = value
                    counter += 1
            else:
                value = 0

            vector_recolor_by_scalar_rgba(polygon[0], value)
            vector_recolor_by_scalar_rgba(polygon[1], value)
            vector_recolor_by_scalar_rgba(polygon[2], value)
        polygon_projection(polygon[0], polygon[1], polygon[2], screen, size, perspective=perspective)

#

# PyGame Initialization and Statements

pygame.init()
pygame.display.set_caption('Quaternion Rotation')

size = width, height = 500, 500

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

FPS = 60
time = 0

execute = True

#

# Working Space #########################################################

i, j, k = 0, 0, 0

r = 100
r2 = r * sqrt(2) / 2
r3 = r / 2

vector1 = vector(r, 0, 0, [0, 0, 0], [102, 0, 225])
vector2 = vector(r2, r2, 0, [0, 0, 0], [102, 0, 225])
vector3 = vector(0, r, 0, [0, 0, 0], [102, 0, 225])
vector4 = vector(-r2, r2, 0, [0, 0, 0], [102, 0, 225])
vector5 = vector(-r, 0, 0, [0, 0, 0], [102, 0, 225])
vector6 = vector(-r2, -r2, 0, [0, 0, 0], [102, 0, 225])
vector7 = vector(0, -r, 0, [0, 0, 0], [102, 0, 225])
vector8 = vector(r2, -r2, 0, [0, 0, 0], [102, 0, 225])
vector9 = vector(r2, 0, -r2, [0, 0, 0], [102, 0, 225])
vector10 = vector(-r2, 0, r2, [0, 0, 0], [102, 0, 225])
vector11 = vector(r2, 0, r2, [0, 0, 0], [102, 0, 225])
vector12 = vector(-r2, 0, -r2, [0, 0, 0], [102, 0, 225])
vector13 = vector(0, r2, r2, [0, 0, 0], [102, 0, 225])
vector14 = vector(0, -r2, -r2, [0, 0, 0], [102, 0, 225])
vector15 = vector(0, r2, -r2, [0, 0, 0], [102, 0, 225])
vector16 = vector(0, -r2, r2, [0, 0, 0], [102, 0, 225])

vector17 = vector(r3, r3, r2, [0, 0, 0], [102, 0, 225])
vector18 = vector(r3, r3, -r2, [0, 0, 0], [102, 0, 225])
vector19 = vector(-r3, r3, r2, [0, 0, 0], [102, 0, 225])
vector20 = vector(-r3, r3, -r2, [0, 0, 0], [102, 0, 225])

vector21 = vector(r3, -r3, r2, [0, 0, 0], [102, 0, 225])
vector22 = vector(r3, -r3, -r2, [0, 0, 0], [102, 0, 225])
vector23 = vector(-r3, -r3, r2, [0, 0, 0], [102, 0, 225])
vector24 = vector(-r3, -r3, -r2, [0, 0, 0], [102, 0, 225])

vector25 = vector(0, 0, r, [0, 0, 0], [102, 0, 225])
vector26 = vector(0, 0, -r, [0, 0, 0], [102, 0, 225])

camera_reposition(0, 200, 0)
# camera_rotation(tau / 4, 0, 0)
light_source(0, 0, 200, 300, 0.25)

# Working Space #########################################################

# Execute

while execute:
    screen.fill((0, 0, 0))

    time += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            execute = False

    # Working Space #########################################################

    i += 0.005
    j += 0.005
    k += 0.005

    vectors = massive_vector_transform(i, j, k, vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8,
                                       vector9, vector10, vector11, vector12, vector13, vector14, vector15, vector16,
                                       vector17, vector18, vector19, vector20, vector21, vector22, vector23, vector24,
                                       vector25, vector26)

    vectors_grav = gravitation_transform(vectors, mass=10, reverse=False, repulse=False)

    points = points_list_sequence(*vectors_grav[0])

    point_render(screen, size, points, [[1, 2, 11], [2, 11, 17], [2, 17, 3], [3, 17, 13], [3, 13, 4], [4, 13, 19],
                                        [4, 19, 5], [5, 19, 10], [5, 10, 6], [10, 6, 23], [23, 6, 7], [7, 23, 16],
                                        [7, 16, 8], [8, 16, 21], [8, 21, 1], [1, 21, 11],
                                        [1, 2, 9], [2, 9, 18], [2, 18, 3], [3, 18, 15], [3, 15, 4], [4, 15, 20],
                                        [4, 20, 5], [5, 20, 12], [5, 12, 6], [12, 6, 24], [24, 6, 7], [7, 24, 14],
                                        [7, 14, 8], [8, 14, 22], [8, 22, 1], [1, 22, 9],
                                        [25, 13, 19], [25, 19, 10], [25, 10, 23], [25, 23, 16], [25, 16, 21],
                                        [25, 21, 11],
                                        [25, 11, 17], [25, 17, 13],
                                        [26, 15, 20], [26, 20, 12], [26, 12, 24], [26, 24, 14], [26, 14, 22],
                                        [26, 22, 9],
                                        [26, 9, 18], [26, 18, 15]
                                        ], light=True, perspective=True)

    # Working Space #########################################################

    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()

#

#####################
##-- By Milorad -- ##
#####################