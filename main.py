import sys
import numpy as np
import math

# {pos:[x,y,z], look_at_position:[x,y,z],up vector:[x,y,z], s_d:number,  s_w:number }
cam = {}
general = {}
mtls = list()
plns = list()

#[{center:[x,y,z],radius:number,mat_index:number}]
spheres = list()
lights = list()
boxes = list()
width = 500
height = 500
def parsing_scene():
    """A Helper function that defines the program arguments."""
    scene_file=sys.argv[1]
    output_image_name=sys.argv[2]

    if len(sys.argv)>3:
        width=sys.argv[3]
        height=sys.argv[4]

    scene_definition_parser(scene_file)

#every point should be numpy array

def get_args_cam(words):
    cam["pos"] = np.array([words[1],words[2],words[3]])
    cam["look_at_position"] = np.array([words[4],words[5],words[6]])
    cam["up_vector"] = np.array([words[7], words[8], words[9]])
    cam["screen_distance"] = words[10]
    cam["screen_width"] = words[11]


def get_args_set(words):
    general["background_color"]=np.array([words[1],words[2],words[3]])
    general["root_num_of_shadow_rays"] = words[4]
    general["max_recursion"] = words[5]


def get_args_mtl(words):
    mtl = {}
    mtl["diffuse_color"] = np.array([words[1],words[2],words[3]])
    mtl["specular_color"] = np.array([words[4],words[5],words[6]])
    mtl["reflection_color"] = np.array([words[7],words[8],words[9]])
    mtl["shininess"] = words[10]
    mtl["transparency"] = words[11]
    mtls.append(mtl)




def get_args_pln(words):
    pln = {}
    pln["normal"] = np.array([words[1], words[2], words[3]])
    pln["offset"] = words[4]
    pln["material_index"] = words[5]
    plns.append(pln)


def get_args_sphere(words):
    sph = {}
    sph["center"] = np.array([words[1], words[2], words[3]])
    sph["radius"] = words[4]
    sph["material_index"] = words[5]
    spheres.append(sph)


def get_args_lgt(words):
    lgt = {}
    lgt["position"] = np.array([words[1], words[2], words[3]])
    lgt["color"] = np.array([words[4], words[5], words[6]])
    lgt["specular_intensity"] = words[7]
    lgt["shadow intensity"] = words[8]
    lgt["radius"] = words[9]
    lights.append(lgt)


def get_args_box(words):
    box = {}
    box["center"] = np.array([words[1], words[2], words[3]])
    box["scale"] = words[4]
    box["material_index"] = words[5]
    boxes.append(box)


def scene_definition_parser(file_name):
    f = open(file_name, "r")



    for line in f:
        if not(line.startswith("#") or len(line.strip()) == 0):
            words=line.split(" ")
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


def RayCast():
    image = np.zeros((height, width, 3), dtype=np.float32)
    E=cam["pos"]
    Vx = np.array([0, 0, 0])
    Vy = np.array([0, 0, 0])

    # calculate Vx,Vy
    # TODO

    P_0 = np.array([0,0,0])

    #calculate P_0
    # TODO


    for i in range(0,height):
        p=P_0
        for j in range(0,width):
            # TODO
            #ray
            #p=E+t(p-E)
            t=1
            min_object = FindIntersection(E,t,p-E)
            if min_object["min_t"]==sys.maxsize:
                #no intersection object
                # TODO
                pass




def FindIntersection(E,t,V):
    #need to do
    min_t = sys.maxsize
    min_primitive={}
    for sph in spheres:
        t = intersectionSphere(E,V,sph)
        if t>0 and t<min_t:
            min_primitive = sph
            min_t = t

    for pln in plns:
        t = intersectionPln(E,V,pln)
        if t>0 and t<min_t:
            min_primitive = pln
            min_t = t

    for box in boxes:
        t = intersectionBox(E,V,box)
        if t>0 and t<min_t:
            min_primitive = box
            min_t = t
    return {"min_t":min_t,"min_primitive":min_primitive}



def intersectionSphere(E,V,sph):
    O = sph["center"]
    r = sph["radius"]
    L = O - E
    t_ca = sum(L*V)
    if(t_ca)<0:
        return 0
    d_square = sum(L*L) - t_ca*t_ca
    if d_square > r*r:
        return 0
    t_hc = math.sqrt(r*r-d_square)
    t = min(t_ca-t_hc,t_ca+t_hc)
    return t

def intersectionPln(E, V, pln):
    # TODO
    pass


def intersectionBox(E, V, box):
    # TODO
    pass



# [{center:[x,y,z],radius:number,mat_index:number}]










if __name__ == "__main__":
    #parsing_scene()
    a=np.array([1,2,3])
    b=np.array([1,2,3])
    print(sum(a*b))