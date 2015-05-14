# pygrid
A Python class for representing GridLab-D objects with a JSON flavor.

Requirements:

--Python 2.7.6, 3.1+

--Tested and Developed in Linux, Ubuntu 14.4+ -- path strings not yet configured for Windows but everything else should work
--Tested and partially developed in Windows 8

--GridLab-D (hopefully you've got this one)

**Disclaimer** Work in progress by a bunch of research scientists.

pygrid is a Python class for representing GridLab-D objects in JSON fashion. Since there is no clear distinction between which physical grid objects do and do require associated configuration objects, all objects in pygrid are treated equally as in Gridlab. In general, an instance of the pygrid class generates a master JSON object (stored as a dictionary) of the following form:

    {object_type:
        {object_name:
            {name: "object_name"
            attribute1: value
            attribute2: value
            }
        another_object_name:
            {name: "another_object_name"
            attribute1: value
            attribute2: value
            }
    other_object_type:
        {
        }
    }
    
Ideal implementations sort object names for a given object type such that quick editing of attribute values can be performed (and not solely, for example, change all capacitor attributes to value X, but rather, capacitors of name Y or possessing attributes Z)

Currently, many features are hardcoded. Continued development may produce generation functions for each object type, better general reading of GLM files into memory, compatibility for GLM write objects (not currently included) and sorting of master JSON key values when writing GLM files, compatability for machine learning techniques, and web rendering.

Example:

Navigate to the pygrid directory and from command line:
    $python compile_123_glm.py
    
This will write the GLM file for the 123 node GLM data stored in IEEE_123_TestFeeder. Additionally, one can call pickle_123_glm.py and this will generated a serialized dictionary, 123IEEE_glm_obj.pck which can be loaded directly into memory.

These scripts should serve as a temporary example for class usage until it can be replaced with something more complete.

In the pygrid directory, once having generating the serialized glm file using 123IEEE_glm_object.pck, one can try calling:
    $python percentload.py 0.85
    
This will generate a similarly named GLM file at 85% nominal load as specified in the IEEE 123 Node Test Feeder documentation.
    
