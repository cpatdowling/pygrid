import pickle, sys
from gridlab_obj import *

def modify_load(load, modifyValue):
    loadVals = {}
    for key in load.keys():
        if key[0:8] == "constant":
            loadVals[key] = load[key].split("+")
            loadVals[key][1] = float(loadVals[key][1].rstrip("j"))
            loadVals[key][0] = float(loadVals[key][0])
    for loadkey in loadVals.keys():
        loadType = loadkey.split("_")[1]
        if loadType == "power":
            loadVals[loadkey][0] = loadVals[loadkey][0]*modifyValue
            loadVals[loadkey][1] = loadVals[loadkey][1]*modifyValue
        if loadType == "current":
            loadVals[loadkey][0] = loadVals[loadkey][0]*modifyValue
            loadVals[loadkey][1] = loadVals[loadkey][1]*modifyValue
        if loadType == "impedance":
            loadVals[loadkey][0] = loadVals[loadkey][0]/modifyValue
            loadVals[loadkey][1] = loadVals[loadkey][1]/modifyValue
        load[loadkey] = str(loadVals[loadkey][0]) + "+" + str(loadVals[loadkey][1]) + "j"
    return(load)

loadratio = float(sys.argv[1])
path = '/home/chase/classes/energy_grids/project/python'
pickleFile = open('123IEEE_glm_obj.pck', 'r')
gl = gridlabObject(filename="123IEEE_" + str(loadratio) + "_load.glm", workingDirectory=path,
header_file="123nodeglmheader.txt")
gl.objects = pickle.load(pickleFile)
pickleFile.close()

loads = gl.objects['load'].keys() 
for load in loads:
    gl.objects['load'][load] = modify_load(gl.objects['load'][load], loadratio)
outPickleFile = open('123IEEE_' + str(loadratio) + '_glm_obj.pck', 'w')
pickle.dump(gl.objects, outPickleFile)
outPickleFile.close()
    
for typeObj in gl.objects.keys():
    for obj in gl.objects[typeObj].keys():
        if obj == "node350":
            pass #we're skipping this node for some reason
        if obj == "node610":
            gl.objects[typeObj][obj]["nominal_voltage"] = "277.1"
        outline = gl.object_string(gl.objects[typeObj][obj], typeObj)
        gl.outFile.write(outline + "\n")
gl.outFile.write("object voltdump {\n\tfilename Volt_Dump_FBS_" + str(loadratio) + ".csv;\n}\n")
gl.outFile.close()
