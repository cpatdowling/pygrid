from gridlab_obj import *

if __name__ == '__main__':
    path = '/home/chase/classes/energy_grids/project/python'
    gl = gridlabObject(filename="123IEEE_python.glm", workingDirectory=path, header_file="123nodeglmheader.txt")
    hardFiles = ['123nodeglmconductors.txt', '123nodeglmlinespacing.txt', '123nodeglmlineconfig.txt', '123nodeglmlines.txt', '123nodeglmloads.txt', '123nodeglmnodes.txt', '123nodeglmtransformer.txt', '123nodeglmcapacitors.txt', '123nodeglmregulators.txt']
    for filename in hardFiles:
        newObjs = gl.read_glm_file(filename)
        print(filename)
        for objType in newObjs.keys():
            if objType not in gl.objects.keys():
                gl.objects[objType] = {}
            for obj in newObjs[objType].keys():
                gl.objects[objType][obj] = newObjs[objType][obj]
    for typeObj in gl.objects.keys():
        allobjects = sorted(gl.objects[typeObj].keys())
        for obj in gl.objects[typeObj].keys():
            if obj == "node350":
                pass #we're skipping this node for some reason
            if obj == "node610":
                gl.objects[typeObj][obj]["nominal_voltage"] = "277.1"
            outline = gl.object_string(gl.objects[typeObj][obj], typeObj)
            gl.outFile.write(outline + "\n")
    gl.outFile.close()

    



"""    
    current = ""
    for line in inFile:
        if gl.check_header(line):
            current = gl.check_header(line)
        elif line[0:4] == "Name":
            pass
        elif current == "Over":
            tokens = line.strip().split(",")
            if "line_configuration" not in gl.reference.keys():
                gl.reference["line_configuration"] = {}
            gl.reference["line_configuration"]["lineconfiguration" + tokens[0]] = {"name": "lineconfiguration" + tokens[0], "phases": "".join(sorted(tokens[1].split(" "))), "spacing": tokens[4], "phase_cond": tokens[2], "neutral_cond": tokens[3]}
        elif current == "Unde":
            tokens = line.strip().split(",")
            if "line_configuration" not in gl.reference.keys():
                gl.reference["line_configuration"] = {}
            gl.reference["line_configuration"]["lineconfiguration" + tokens[0]] = {"name": "lineconfiguration" + tokens[0], "phases": "".join(tokens[1].split(" ")), "spacing": tokens[3]}
        elif current == "Line":
            tokens = line.strip().split(",")
            name = "node" + tokens[0] + "node" + tokens[1]
            #need to recover phases
            config = gl.objects["line_configuration"]["lineconfiguration" + tokens[3]]
            configname = config["name"]
            linephases = gl.reference["line_configuration"]["lineconfiguration" + tokens[3]]["phases"]
            obj = gl.create_line(name, linephases, "node" + tokens[0], "node" + tokens[1], tokens[2], configname)
            if obj["object"] not in gl.objects.keys():
                gl.objects[obj["object"]] = {}
            gl.objects[obj["object"]][name] = obj
        elif current == "Spot":
            tokens = line.strip().split(",")
            name = "node" + tokens[0]
            mode = gl.get_load_mode(tokens[1])
            phases = gl.get_phases_from_vals_node(tokens[2:], mode, 4160.0)
            phasevals = {}
            for key in phases.keys():
                phasevals[mode + "_" + key] = phases[key]
            obj = gl.create_load(name, phasevals)
            if obj["object"] not in gl.objects.keys():
                gl.objects[obj["object"]] = {}
            gl.objects[obj["object"]][name] = obj
"""
