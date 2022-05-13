import sys
import numpy as np
import math
import os
from typing import Dict, Any

from PIL import Image
from parsing import cam, general, mtls, plns, spheres, lights, boxes
from parsing import scene_definition_parser
import random

NDArray = Any
Options = Any
scene_file = None
output_image_name = ""

width = 500
height = 500


# returns min t of instersection of shadow ray.
# if no intersection returns 0
# @param E :point that we shoot the ray from
# @param V : direction of the ray
# @param t_intersection : the original t intersection we calculate light intensity for
def FindIntersection_shadow(E, V, t_intersection):
    min_t = np.inf
    type_p = ""
    min_primitive = {}
    epsilon = pow(10, -6)
    for sph in spheres:
        t = intersectionSphere(E, V, sph)
        if 0 < t < t_intersection and not abs(t - t_intersection) <= epsilon:
            # if t<t_intersection and not abs(t-t_intersection)<=epsilon:
            return 0

    for pln in plns:
        t = intersectionPln(E, V, pln)
        if 0 < t < t_intersection and not abs(t - t_intersection) <= epsilon:
            # if t<t_intersection and not abs(t-t_intersection)<=epsilon:
            return 0

    for box in boxes:
        t = intersectionBox(E, V, box)
        if 0 < t < t_intersection and not abs(t - t_intersection) <= epsilon:
            # if t<t_intersection and not abs(t-t_intersection)<=epsilon:
            return 0
    return 1


# returns min t of intersection with sphere
# if no intersection returns 0
# @param E :point that we shoot the ray from
# @param V : direction of the ray
# @param sph : the sphere we intersect with
def intersectionSphere(E, V, sph):
    O = sph["center"]
    r = sph["radius"]
    L = O - E
    t_ca = sum(L * V)
    if (t_ca) < 0:
        return 0
    d_square = sum(L * L) - t_ca * t_ca
    if d_square > r * r:
        return 0
    t_hc = math.sqrt(r * r - d_square)
    t = min(t_ca - t_hc, t_ca + t_hc)
    return t


# returns min t of intersection with pln
# if no intersection returns 0
# @param E :point that we shoot the ray from
# @param V : direction of the ray
# @param pln : the pln we intersect with
# pln:{"normal":[x,y,z],"offset":number}
def intersectionPln(E, V, pln):
    N = pln["normal"]
    c = pln["offset"]
    if sum(V * N) == 0:
        return 0
    t = (-1) * (sum(E * N) - c) / sum(V * N)
    return t


# returns min t of intersection with pln
# if no intersection returns 0
# @param E :point that we shoot the ray from
# @param V : direction of the ray
# @param box : the box we intersect with
# box: {center:[x,y,z],scale:number,"material_index":number}
def intersectionBox(E, V, box):
    t_near = -np.inf
    t_far = np.inf
    center_box = box["center"]
    scale = box["scale"]
    x1 = min(center_box[0] + scale / 2, center_box[0] - scale / 2)
    x2 = max(center_box[0] + scale / 2, center_box[0] - scale / 2)

    y1 = min(center_box[1] + scale / 2, center_box[1] - scale / 2)
    y2 = max(center_box[1] + scale / 2, center_box[1] - scale / 2)

    z1 = min(center_box[2] + scale / 2, center_box[2] - scale / 2)
    z2 = max(center_box[2] + scale / 2, center_box[2] - scale / 2)

    one = [x1, y1, z1]
    two = [x2, y2, z2]

    for i in range(3):
        if V[i] == 0:
            if E[i] < one[i] or E[i] > two[i]:
                return 0
        else:
            t1 = (one[i] - E[i]) / V[i]
            t2 = (two[i] - E[i]) / V[i]

            if t1 > t2:
                # swap t1 and t2
                temp = t1
                t1 = t2
                t2 = temp

            if t1 > t_near:
                t_near = t1

            if t2 < t_far:
                t_far = t2

            if t_near > t_far:
                return 0

            if t_far < 0:
                return 0
    return t_near


# returns min intersection {"min_t": min_t, "min_primitive": min_primitive, "type": type_p}  with ray
# if no intersection returns {"min_t": np.inf, "min_primitive": {}, "type": ""}
# @param E :point that we shoot the ray from
# @param V : direction of the ray
def FindIntersection(E, V):
    min_t = np.inf
    type_p = ""
    min_primitive = {}
    for sph in spheres:
        t = intersectionSphere(E, V, sph)
        if 0 < t < min_t:
            min_primitive = sph
            min_t = t
            type_p = "sph"

    for pln in plns:
        t = intersectionPln(E, V, pln)
        if 0 < t < min_t:
            min_primitive = pln
            min_t = t
            type_p = "pln"

    for box in boxes:
        t = intersectionBox(E, V, box)
        if 0 < t < min_t:
            min_primitive = box
            min_t = t
            type_p = "box"
    return {"min_t": min_t, "min_primitive": min_primitive, "type": type_p}


'''
def calculate_M(a, b, c):
    Sx = -b
    Cx = math.sqrt(1 - Sx * Sx)
    Sy = (-a) / Cx
    Cy = c / Cx
    return {"Sx": Sx, "Cx": Cx, "Sy": Sy, "Cy": Cy}
'''


# returns the I_diff of intersection point of ray
# @param N = the surface normal
# @param L = vector light direction
# @param I_p = light intensity number
# @param K_d = [R,G,B] diffuse surface color
def calculate_I_diff(N, L, I_p, K_d):
    dot_product = max(0, sum(N * L))
    result = dot_product * I_p * K_d
    result = [min(x, 1) for x in result]
    result = [max(x, 0) for x in result]
    return result


# returns the I_spec of intersection point of ray
# @param K_s = [R,G,B] specular surface color
# @param I_p = light intensity number
# @param R = reflection vector of the light
# @param V = vector from the camera to intersection point
# @param n = specular reflection parameter
def calculate_Ipec(K_s, I_p, R, V, n):
    dot_product = max(-sum(R * V), 0)
    return math.pow(dot_product, n) * I_p * K_s

#returns the normalized vector of v
def normalize(v):
    return v / math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

#returns normal of a vector
def normal(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

#returns normal for the point on a box
def calculate_normal_box(box, P):
    epsilon = pow(10, -6)
    center_box = box["center"]
    radius_box = box["scale"]

    front_mid = np.array([center_box[0], center_box[1], center_box[2] + radius_box / 2])
    back_mid = np.array([center_box[0], center_box[1], center_box[2] - radius_box / 2])

    right_mid = np.array([center_box[0] + radius_box / 2, center_box[1], center_box[2]])
    left_mid = np.array([center_box[0] - radius_box / 2, center_box[1], center_box[2]])

    up_mid = np.array([center_box[0], center_box[1] + radius_box / 2, center_box[2]])
    down_mid = np.array([center_box[0], center_box[1] - radius_box / 2, center_box[2]])

    if abs(P[2] - front_mid[2]) <= epsilon:
        return normalize(front_mid - center_box)
    if abs(P[2] - back_mid[2]) <= epsilon:
        return normalize(back_mid - center_box)

    if abs(P[0] - right_mid[0]) <= epsilon:
        return normalize(right_mid - center_box)

    if abs(P[0] - left_mid[0]) <= epsilon:
        return normalize(left_mid - center_box)

    if abs(P[1] - up_mid[1]) <= epsilon:
        return normalize(up_mid - center_box)
    return normalize(down_mid - center_box)

#returns color of the min intersection point of a ray
# @param E :point that we shoot the ray from
# @param V : direction of the ray
# @param t : offset on the ray V(the intersection point)
# @param primitive : dictionary of the object of the intersection point
# @param type : type of the primitive. can be "box","pln","sph".
# @param recursion_level : in what recursion level we are at.
def calculate_color(E, V, t, primitive, type, recursion_level):
    # print("rec: "+str(recursion_level))
    if recursion_level > general["max_recursion"]:
        return general["background_color"]

    P = E + t * V
    color = np.array([0, 0, 0])
    diffuse_color = np.array([0, 0, 0])
    spec_color = np.array([0, 0, 0])
    reflection_color = np.array([0, 0, 0])
    background_color = general["background_color"]
    transperancy_mtl = mtls[primitive["material_index"] - 1]["transparency"]
    primitive_diffuse_color = mtls[primitive["material_index"] - 1]["diffuse_color"]
    primitive_spec_color = mtls[primitive["material_index"] - 1]["specular_color"]
    primitive_reflection_color = mtls[primitive["material_index"] - 1]["reflection_color"]
    n = mtls[primitive["material_index"] - 1]["shininess"]
    N = np.array([0, 0, 0])
    if type == "sph":
        N = normalize(P - primitive["center"])

    if type == "pln":
        pln_normal_normalize = normalize(primitive["normal"])
        N = pln_normal_normalize
    if type == "box":
        N = calculate_normal_box(primitive, P)

    for light in lights:

        L = normalize(light["position"] - P)
        I_p = float(soft_shadows(light, general["root_num_of_shadow_rays"], P))
        # I_p=1
        K_d = primitive_diffuse_color
        I_diff = calculate_I_diff(N, L, I_p, K_d)
        diffuse_color = diffuse_color + I_diff * light["color"]

        R = (sum(2 * L * N)) * N - L
        Ks = primitive_spec_color
        I_spec = calculate_Ipec(Ks, I_p, R, V, n)
        spec_color = spec_color + light["specular_intensity"] * I_spec * light["color"]

    R = V - 2 * (sum(V * N)) * N
    next_primitive = FindIntersection(P, normalize(R))
    if not math.isinf(next_primitive["min_t"]):
        # TODO what if the intersection is the same primitive?

        color_from_reflection = calculate_color(P, normalize(R), next_primitive["min_t"],
                                                next_primitive["min_primitive"],
                                                next_primitive["type"], recursion_level + 1)
        reflection_color = reflection_color + color_from_reflection * primitive_reflection_color

    else:
        reflection_color = background_color * primitive_reflection_color
    diffuse_color = [min(x, 1) for x in diffuse_color]
    diffuse_color = [max(x, 0) for x in diffuse_color]
    color = background_color * transperancy_mtl + (1 - transperancy_mtl) * (
            diffuse_color + spec_color) + reflection_color
    color = [min(x, 1) for x in color]
    color = [max(x, 0) for x in color]

    return color

'''
def equal_array(arr1, arr2):
    epsilon = pow(10, -6)
    if abs(arr1[0] - arr2[0]) <= epsilon and abs(arr1[1] - arr2[1]) <= epsilon and abs(arr1[2] - arr2[2]) <= epsilon:
        return True
    return False
'''

#calculates the soft shadows for intersection point. returns light intensity of the point.
# @param light :dictionary of the light
# @param N :normal to the surface
# @param intersection_point the intersection point we calculate the light intensity for.
def soft_shadows(light, N, intersection_point):


    sum_hit_rays = 0
    P = light["position"]
    Vz = normalize(intersection_point - P)
    up_input = np.array([0, 1, 0])
    right = normalize(np.cross(up_input, Vz))
    up_vector = normalize(np.cross(Vz, right))
    Vy = up_vector
    Vx = right
    # M = calculate_M(Vz[0], Vz[1], Vz[2])
    # Vx = np.array([M["Cy"], 0, M["Sy"]])
    # Vy = np.array([-M["Sx"] * M["Sy"], M["Cx"], M["Sx"] * M["Cy"]])

    radius = light["radius"]
    P_0 = P - float(radius / 2) * Vx - float(radius / 2) * Vy

    for i in range(0, N):
        p_curr = P_0
        for j in range(0, N):
            random_x = random.random()
            random_y = random.random()
            p_random = p_curr + (radius / N) * random_x * Vx + (radius / N) * random_y * Vy

            ray = normalize(intersection_point - p_random)
            t_inter = normal(intersection_point - p_random)
            sum_hit_rays = sum_hit_rays + FindIntersection_shadow(p_random, ray, t_inter)
            '''
            min_object = FindIntersection(p_random, normalize(intersection_point-p_random))
            t = min_object["min_t"]
            curr_intersection = p_random+normalize(intersection_point-p_random  )*t
            if equal_array(intersection_point,curr_intersection):
                sum_hit_rays = sum_hit_rays+1
            '''
            p_curr = p_curr + Vx * (radius / N)
        P_0 = P_0 + Vy * (radius / N)
    # print("sum: "+str(sum_hit_rays))
    percent_hit_rays = sum_hit_rays / (float(N) * N)

    shadow_intensity = light["shadow_intensity"]
    light_intensity = 1 * (1 - shadow_intensity) + shadow_intensity * percent_hit_rays
    # print(light_intensity)
    return light_intensity

#main function of ray casting.
#returns the final image.
def RayCast():
    image = np.zeros((height, width, 3), dtype=np.float64)
    look_at_point = cam["look_at_position"]
    E = cam["pos"]
    f = cam["screen_distance"]
    up_input = cam["up_vector"]
    P = E + f * normalize(look_at_point - E)

    Vz = (P - E) / f
    right = normalize(np.cross(up_input, Vz))
    fixed_up_vector = normalize(np.cross(right, Vz))
    if sum(fixed_up_vector * up_input) > 0:
        up_vector = fixed_up_vector
    else:
        up_vector = fixed_up_vector * (-1)
    Vy = up_vector
    Vx = right
    # M = calculate_M(Vz[0], Vz[1], Vz[2])
    # Vx = np.array([M["Cy"], 0, M["Sy"]])
    # Vy = np.array([-M["Sx"] * M["Sy"], M["Cx"], M["Sx"] * M["Cy"]])

    width_screen = cam["screen_width"]
    ratio = float(width) / height
    height_screen = width_screen / ratio

    P_0 = P - float(width_screen / 2) * Vx - float(height_screen / 2) * Vy

    for i in range(height - 1, -1, -1):
        p = P_0
        for j in range(0, width):

            min_object = FindIntersection(E, normalize(p - E))
            if math.isinf(min_object["min_t"]):
                image[i][j] = general["background_color"]
            else:
                image[i][j] = calculate_color(E, normalize(p - E), min_object["min_t"], min_object["min_primitive"],
                                              min_object["type"], 0)

            p = p + Vx * (width_screen / width)
            print("i : " + str(i) + " j : " + str(j))
        P_0 = P_0 + Vy * (height_screen / height)
    return image

#parsing the scene and update the global fields of the scene.
def parsing_scene():
    """A Helper function that defines the program arguments."""

    global scene_file
    global output_image_name
    scene_file = sys.argv[1]
    output_image_name = sys.argv[2]

    if len(sys.argv) > 3:
        global width
        global height
        width = int(sys.argv[3])
        height = int(sys.argv[4])

    scene_definition_parser(scene_file)


def normalize_image(image: NDArray):
    """Normalize image pixels to be between [0., 1.0]"""
    min_img = image.min()
    max_img = image.max()
    # normalized_image = (image - min_img) / (max_img - min_img)
    normalized_image = image * 255
    return normalized_image


def save_image(image: NDArray, image_loc: str):
    """A helper method that saves a dictionary of images"""

    def _prepare_to_save(image: NDArray):
        """Helper method that converts the image to Uint8"""
        if image.dtype == np.uint8:
            return image
        return normalize_image(image).astype(np.uint8)

    Image.fromarray(_prepare_to_save(image)).save(f'{image_loc}')


if __name__ == "__main__":
    import time

    start_time = time.time()
    parsing_scene()
    image = RayCast()
    # image = np.zeros((100, 100, 3), dtype=np.float64)
    # image[0][0]=[1,0,0]
    save_image(image, output_image_name)
    print("--- %s seconds ---" % (time.time() - start_time))
    # a = np.array([1, 2, 3])
    # b = np.array([1, 2, 3])
    # print(float(1 / 3) * a)
