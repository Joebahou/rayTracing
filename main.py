import argparse
import sys
import numpy as np
import math

# {pos:[x,y,z], look_at_position:[x,y,z],up vector:[x,y,z], s_d:number,  s_w:number }
cam = {}
general = {}
mtls = []
plns = []

#[{center:[x,y,z],radius:number,mat_index:number}]
spheres = []
lights = []
boxes = []
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

def get_args_cam(words, cam):
    pass


def get_args_set(words, general):
    pass


def get_args_mtl(words, mtl):
    pass


def get_args_pln(words, pln):
    pass


def get_args_sphere(words, sphere):
    pass


def get_args_lgt(words, lgt):
    pass


def get_args_box(words, box):
    pass


def scene_definition_parser(file_name):
    f = open(file_name, "r")



    for line in f:
        if not(line.startswith("#") or len(line.strip()) == 0):
            words=line.split(" ")
            cases = {
                "cam": lambda: get_args_cam(words, cam),
                "set": lambda: get_args_set(words, general),
                "mtl": lambda: get_args_mtl(words, mtl),
                "pln": lambda: get_args_pln(words, pln),
                "sph": lambda: get_args_sphere(words, sphere),
                "lgt": lambda: get_args_lgt(words, lgt),
                "box": lambda: get_args_box(words, box)
            }
            cases.get(words[0].strip(), lambda: print("Didn't match a case"))()




    f.close()


def RayCast():
    image = np.zeros((height, width, 3), dtype=np.float32)
    E=cam["pos"]
    Vx = np.array([0, 0, 0])
    Vy = np.array([0, 0, 0])

    # calculate Vx,Vy

    P_0 = np.array([0,0,0])

    #calculate P_0


    for i in range(0,height):
        p=P_0
        for j in range(0,width):
            #ray
            #p=E+t(p-E)
            t=1
            min_object = FindIntersection(E,t,p-E)
            if min_object["min_t"]==sys.maxsize:
                #no intersection object
                pass


def FindIntersection(E,t,V):
    #need to do
    min_t = sys.maxsize
    min_primitive={}
    for sph in spheres:
        t = IntersectionSphere(E,V,sph)
        if t>0 and t<min_t:
            min_primitive = sph
            min_t = t

    for pln in plns:
        t = IntersectionPln(E,V,pln)
        if t>0 and t<min_t:
            min_primitive = pln
            min_t = t

    for box in boxes:
        t = IntersectionBox(E,V,box)
        if t>0 and t<min_t:
            min_primitive = box
            min_t = t
    return {"min_t":min_t,"min_primitive":min_primitive}



def IntersectionSphere(E,V,sph):
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



# [{center:[x,y,z],radius:number,mat_index:number}]










if __name__ == "__main__":
    #parsing_scene()
    a=np.array([1,2,3])
    b=np.array([1,2,3])
    print(sum(a*b))