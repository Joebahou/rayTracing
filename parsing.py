import numpy as np


cam = {}
general = {}
mtls = list()
plns = list()
index_primitive = 0
# [{center:[x,y,z],radius:number,mat_index:number}]
spheres = list()
lights = list()
boxes = list()


# every point should be numpy array

def get_args_cam(words):
    cam["pos"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    cam["look_at_position"] = np.array([float(words[4].strip()), float(words[5].strip()), float(words[6].strip())])
    cam["up_vector"] = np.array([float(words[7].strip()), float(words[8].strip()), float(words[9].strip())])
    cam["screen_distance"] = float(words[10].strip())
    cam["screen_width"] = float(words[11].strip())


def get_args_set(words):
    general["background_color"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    general["root_num_of_shadow_rays"] = int(words[4].strip())
    general["max_recursion"] = int(words[5].strip())


def get_args_mtl(words):
    mtl = {}
    mtl["diffuse_color"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    mtl["specular_color"] = np.array([float(words[4].strip()), float(words[5].strip()), float(words[6].strip())])
    mtl["reflection_color"] = np.array([float(words[7].strip()), float(words[8].strip()), float(words[9].strip())])
    mtl["shininess"] = float(words[10].strip())
    mtl["transparency"] = float(words[11].strip())
    mtls.append(mtl)


def get_args_pln(words):
    pln = {}
    pln["normal"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    pln["offset"] = float(words[4].strip())
    pln["material_index"] = int(words[5].strip())
    global index_primitive
    pln["index"]=index_primitive
    index_primitive=index_primitive+1
    plns.append(pln)


def get_args_sphere(words):
    sph = {}
    sph["center"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    sph["radius"] = float(words[4].strip())
    sph["material_index"] = int(words[5].strip())
    global index_primitive
    sph["index"]=index_primitive
    index_primitive = index_primitive + 1
    spheres.append(sph)


def get_args_lgt(words):
    lgt = {}
    lgt["position"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    lgt["color"] = np.array([float(words[4].strip()), float(words[5].strip()), float(words[6].strip())])
    lgt["specular_intensity"] = float(words[7].strip())
    lgt["shadow_intensity"] = float(words[8].strip())
    lgt["radius"] = float(words[9].strip())
    lights.append(lgt)


def get_args_box(words):
    box = {}
    box["center"] = np.array([float(words[1].strip()), float(words[2].strip()), float(words[3].strip())])
    box["scale"] = float(words[4].strip())
    box["material_index"] = int(words[5].strip())
    global index_primitive
    box["index"]=index_primitive
    index_primitive = index_primitive + 1
    boxes.append(box)


def scene_definition_parser(file_name):
    f = open(file_name, "r")
    index_primitive=0

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
