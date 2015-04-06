import sys, os, pickle
from gridlab_obj import *

if __name__ == '__main__':
    path = '/home/chase/classes/energy_grids/project/python'
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
pickleFile = open(path + "/123IEEE_glm_obj.pck", 'w')
pickle.dump( gl.objects, pickleFile )
pickleFile.close()
