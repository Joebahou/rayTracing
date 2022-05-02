import numpy as np


cam = {}
general = {}
mtls = list()
plns = list()

# [{center:[x,y,z],radius:number,mat_index:number}]
spheres = list()
lights = list()
boxes = list()


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
        line=line.strip()

        if not (line.startswith("#") or len(line.strip()) == 0):
            words = line.replace("\t"," ")
            words = words.split(" ")
            input_line=list()
            for word in words:
                if len(word.strip()) != 0:
                    input_line.append(word)
            cases = {
                "cam": lambda: get_args_cam(input_line),
                "set": lambda: get_args_set(input_line),
                "mtl": lambda: get_args_mtl(input_line),
                "pln": lambda: get_args_pln(input_line),
                "sph": lambda: get_args_sphere(input_line),
                "lgt": lambda: get_args_lgt(input_line),
                "box": lambda: get_args_box(input_line)
            }
            cases.get(input_line[0], lambda: print("Didn't match a case"))()

    f.close()
