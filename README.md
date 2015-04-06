# pygrid
A Python class for representing GridLab-D objects with a JSON flavor.

Requirements: 
    Python 2.7.6, 3.1+
    Tested and Developed in linux, Ubuntu 14.4 -- path strings not yet configured for Windows but everything else should work
    GridLab-D (hopefully you've got this one)

**Disclaimer** Work in progress by a bunch of research scientists.

pygrid is a Python class for representing GridLab-D objects in JSON fashion. Having no authority, this developer does not pass judgement on GridLab-D engineer's choice of what jbjects did and did not require configuration objects, so all objects in GridLab-D are treated equally. In general, an instance of the pygrid class generates a master JSON object (stored as a dictionary) of the following form:

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
    
Ideal implementations sort object names for a given object type such that quick editing of attribute values can be performed (and not solely, for example, change all capacitor attributes to value X, but rather, capacitors of name Y or possesing attributes Z)

Currently, many features are hardcoded. Continued development may produce generation functions for each object type, better general reading of GLM files into memory, compatability for GLM write objects (not currently included) and sorting of master JSON key values when writing GLM files
    
