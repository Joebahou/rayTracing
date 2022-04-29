import sys
import numpy as np
import math

# {pos:[x,y,z], look_at_position:[x,y,z],up vector:[x,y,z], s_d:number,  s_w:number }
cam = {}
general = {}
mtls = list()
plns = list()

# [{center:[x,y,z],radius:number,mat_index:number}]
spheres = list()
lights = list()
boxes = list()
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

#pln:{"normal":[x,y,z],"offset":number}
def intersectionPln(E, V, pln):
    N = pln["normal"]
    c = pln["offset"]
    t = (-1)*(sum(E*N)-c) / sum(V*N)
    return t




def intersectionBox(E, V, box):
    # TODO
    return 0


def FindIntersection(E, t, V):
    # need to do
    min_t = sys.maxsize
    min_primitive = {}
    for sph in spheres:
        t = intersectionSphere(E, V, sph)
        if 0 < t < min_t:
            min_primitive = sph
            min_t = t

    for pln in plns:
        t = intersectionPln(E, V, pln)
        if t > 0 and t < min_t:
            min_primitive = pln
            min_t = t

    for box in boxes:
        t = intersectionBox(E, V, box)
        if t > 0 and t < min_t:
            min_primitive = box
            min_t = t
    return {"min_t": min_t, "min_primitive": min_primitive}


def calculate_M(a, b, c):
    Sx = -b
    Cx = math.sqrt(1 - math.sqrt(Sx))
    Sy = (-a) / Cx
    Cy = c / Cx
    return {"Sx": Sx, "Cx": Cx, "Sy": Sy, "Cy": Cy}


def RayCast():
    image = np.zeros((height, width, 3), dtype=np.float32)
    P = np.array([0, 0, 0])

    # calculate P(middle of the screen)
    # TODO

    E = cam["pos"]
    f = cam["screen_distance"]
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
            min_object = FindIntersection(E, t, p - E)
            if min_object["min_t"] == sys.maxsize:
                # no intersection object
                # TODO
                pass

            #update image
            # TODO

            p = p + Vx
        P_0 = P_0 + Vy
    return image



# every point should be numpy array

def get_args_cam(words):
    cam["pos"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    cam["look_at_position"] = np.array([words[4].strip(), words[5].strip(), words[6].strip()])
    cam["up_vector"] = np.array([words[7].strip(), words[8].strip(), words[9].strip()])
    cam["screen_distance"] = words[10].strip()
    cam["screen_width"] = words[11].strip()


def get_args_set(words):
    general["background_color"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    general["root_num_of_shadow_rays"] = words[4].strip()
    general["max_recursion"] = words[5].strip()


def get_args_mtl(words):
    mtl = {}
    mtl["diffuse_color"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    mtl["specular_color"] = np.array([words[4].strip(), words[5].strip(), words[6].strip()])
    mtl["reflection_color"] = np.array([words[7].strip(), words[8].strip(), words[9].strip()])
    mtl["shininess"] = words[10].strip()
    mtl["transparency"] = words[11].strip()
    mtls.append(mtl)


def get_args_pln(words):
    pln = {}
    pln["normal"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    pln["offset"] = words[4].strip()
    pln["material_index"] = words[5].strip()
    plns.append(pln)


def get_args_sphere(words):
    sph = {}
    sph["center"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    sph["radius"] = words[4].strip()
    sph["material_index"] = words[5].strip()
    spheres.append(sph)


def get_args_lgt(words):
    lgt = {}
    lgt["position"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    lgt["color"] = np.array([words[4].strip(), words[5].strip(), words[6].strip()])
    lgt["specular_intensity"] = words[7].strip()
    lgt["shadow intensity"] = words[8].strip()
    lgt["radius"] = words[9].strip()
    lights.append(lgt)


def get_args_box(words):
    box = {}
    box["center"] = np.array([words[1].strip(), words[2].strip(), words[3].strip()])
    box["scale"] = words[4].strip()
    box["material_index"] = words[5].strip()
    boxes.append(box)


def scene_definition_parser(file_name):
    f = open(file_name, "r")

    for line in f:
        if not (line.startswith("#") or len(line.strip()) == 0):
            words = line.split(" ")
            cases = {
                "cam": lambda: get_args_cam(words),
                "set": lambda: get_args_set(words),
                "mtl": lambda: get_args_mtl(words),
                "pln": lambda: get_args_pln(words),
                "sph": lambda: get_args_sphere(words),
                "lgt": lambda: get_args_lgt(words),
                "box": lambda: get_args_box(words)
            }
            cases.get(words[0].strip(), lambda: print("Didn't match a case"))()

    f.close()


def parsing_scene():
    """A Helper function that defines the program arguments."""
    scene_file = sys.argv[1]
    output_image_name = sys.argv[2]

    if len(sys.argv) > 3:
        # TODO
        #change the global width,height

        width = sys.argv[3]
        height = sys.argv[4]

    scene_definition_parser(scene_file)


if __name__ == "__main__":
    # parsing_scene()
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    print(float(1 / 3) * a)
