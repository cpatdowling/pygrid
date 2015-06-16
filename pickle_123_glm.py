import os
import pickle
import sys
from pygrid import *

if __name__ == '__main__':
    pygridpath = os.getcwd()
    path = os.path.join(pygridpath, "IEEE_123_TestFeeder")
    gl = gridlabObject(filename="123IEEE_python.glm", workingDirectory=path, header_file="123nodeglmheader.txt")
    hardFiles = ['123nodeglmconductors.txt', '123nodeglmlinespacing.txt', '123nodeglmlineconfig.txt', '123nodeglmlines.txt', '123nodeglmloads.txt', '123nodeglmnodes.txt', '123nodeglmtransformer.txt', '123nodeglmcapacitors.txt', '123nodeglmregulators.txt']
    for filename in hardFiles:
        print(filename)
        newObjs = gl.read_glm_file(filename)
        for objType in newObjs.keys():
            if objType not in gl.objects.keys():
                gl.objects[objType] = {}
            for obj in newObjs[objType].keys():
                gl.objects[objType][obj] = newObjs[objType][obj]
    pickleFilePath = os.path.join(pygridpath, "123IEEE_glm_obj.pck")
    with open(pickleFilePath, 'wb') as pickleFile:
        pickle.dump( gl.objects, pickleFile )