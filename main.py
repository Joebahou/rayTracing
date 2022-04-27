import argparse
import sys
import numpy as np

def parsing_scene():
    """A Helper function that defines the program arguments."""
    scene_file=sys.argv[1]
    output_image_name=sys.argv[2]
    width = 500
    height = 500
    if len(sys.argv)>3:
        width=sys.argv[3]
        height=sys.argv[4]

    scene_definition_parser(scene_file)


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
    # {pos:[x,y,z], look-at position:[x,y,z],up vector:[x,y,z], s_d:number,  s_w:number
    cam = {}
    general = {}
    mtl = []
    pln = []
    sphere = []
    lgt = []
    box = []

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











if __name__ == "__main__":
    parsing_scene()