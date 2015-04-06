from gridlab_obj import *

if __name__ == "__main__":
    path = '/home/chase/classes/energy_grids/project/python'
    gl = gridlabObject()
    hardFiles = ['123nodeglmconductors.txt', '123nodeglmlinespacing.txt', '123nodeglmlineconfig.txt', '123nodeglmtransformer.txt']
    for filename in hardFiles:
        newObjs = gl.read_glm_file(filename)
        print(newObjs)
        for objType in newObjs.keys():
            if objType not in gl.objects.keys():
                gl.objects[objType] = {}
            for obj in newObjs[objType].keys():
                gl.objects[objType][obj] = newObjs[objType][obj]
    print(gl.objects)
    for typeObj in gl.objects.keys():
        for obj in gl.objects[typeObj].keys():
            outline = gl.object_string(gl.objects[typeObj][obj], typeObj)
            gl.outFile.write(outline + "\n")
    gl.outFile.close()    
