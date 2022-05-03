import sys
import numpy as np
import math
import os
from typing import Dict, Any

from PIL import Image
from parsing import cam, general, mtls, plns, spheres, lights, boxes
from parsing import scene_definition_parser

NDArray = Any
Options = Any
scene_file = None
output_image_name = ""
# {pos:[x,y,z], look_at_position:[x,y,z],up vector:[x,y,z], s_d:number,  s_w:number }

width = 500
height = 500


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


# pln:{"normal":[x,y,z],"offset":number}
def intersectionPln(E, V, pln):
    N = pln["normal"]
    c = pln["offset"]
    t = (-1) * (sum(E * N) - c) / sum(V * N)
    return t


# check it is correct
# TODO
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


def FindIntersection(E, t, V):
    # todo: need to do
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


def calculate_M(a, b, c):
    Sx = -b
    Cx = math.sqrt(1 - Sx*Sx)
    Sy = (-a) / Cx
    Cy = c / Cx
    return {"Sx": Sx, "Cx": Cx, "Sy": Sy, "Cy": Cy}


# I_p = light intensity number
# K_d = [R,G,B] diffuse surface color
def calculate_I_diff(N, L, I_p, K_d):
    dot_product = -sum(N * L)
    return dot_product * I_p * K_d


# I_p = light intensity number
# K_s = [R,G,B] specular surface color
def calculate_Ipec(K_s, I_p, R, V, n):
    dot_product = -sum(R * V)
    return math.pow(dot_product, n) * I_p * K_s


def normalize(v):
    return v / math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def calculate_color(E, V, t, primitive, type):
    P = E + t * V
    color = np.array([0, 0, 0])
    primitive_diffuse_color = mtls[primitive["material_index"] - 1]["diffuse_color"]
    primitive_spec_color = mtls[primitive["material_index"] - 1]["specular_color"]
    n = mtls[primitive["material_index"] - 1]["shininess"]
    if type == "sph":
        N = normalize(P - primitive["center"])
        for light in lights:
            # need to check if the light intersect other objects before that
            # soft shadows
            # TODO

            L = normalize(light["position"] - P)
            I_p = 1
            K_d = primitive_diffuse_color
            I_diff = calculate_I_diff(N, L, I_p, K_d)
            color = color+ I_diff * light["color"]
            '''
            R = (2 * L * N) * N - L
            Ks = primitive_spec_color
            I_spec = calculate_Ipec(Ks, I_p, R, V, n)
            color = color + light["specular_intensity"]*light["color"]*I_spec
            '''
    color = [min(x, 1) for x in color]
    color = [max(x, 0) for x in color]
    #return mtls[primitive["material_index"] - 1]["diffuse_color"]
    return color


def RayCast():
    image = np.zeros((height, width, 3), dtype=np.float64)
    look_at_point = cam["look_at_position"]
    E = cam["pos"]
    f = cam["screen_distance"]
    P = E + f * normalize(look_at_point - E)

    Vz = (P - E) / f
    M = calculate_M(Vz[0], Vz[1], Vz[2])
    Vx = np.array([M["Cy"], 0, M["Sy"]])
    Vy = np.array([-M["Sx"] * M["Sy"], M["Cx"], M["Sx"] * M["Cy"]])

    width_screen = cam["screen_width"]
    ratio = float(width) / height
    height_screen = width_screen / ratio

    P_0 = P - float(width_screen / 2) * Vx - float(height_screen / 2) * Vy

    for i in range(0, height):
        p = P_0
        for j in range(0, width):
            # TODO
            # ray
            # p=E+t(p-E)
            t = 1
            min_object = FindIntersection(E, t, normalize(p - E))
            if math.isinf(min_object["min_t"]):
                # no intersection object
                # TODO
                image[i][j] = [0, 0, 0]
            else:
                image[i][j] = calculate_color(E, normalize(p - E), min_object["min_t"], min_object["min_primitive"],
                                              min_object["type"])

            # update image
            # TODO
            p = p + Vx*(width_screen/width)
        P_0 = P_0 + Vy*(height_screen/height)
    return image


def parsing_scene():
    """A Helper function that defines the program arguments."""

    global scene_file
    global output_image_name
    scene_file = sys.argv[1]
    output_image_name = sys.argv[2]

    if len(sys.argv) > 3:
        # TODO
        # change the global width,height
        global width
        global height
        width = int(sys.argv[3])
        height = int(sys.argv[4])

    scene_definition_parser(scene_file)


def normalize_image(image: NDArray):
    """Normalize image pixels to be between [0., 1.0]"""
    min_img = image.min()
    max_img = image.max()
    #normalized_image = (image - min_img) / (max_img - min_img)
    normalized_image = image*255
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
    parsing_scene()
    image = RayCast()
    save_image(image, output_image_name)
    # a = np.array([1, 2, 3])
    # b = np.array([1, 2, 3])
    # print(float(1 / 3) * a)
