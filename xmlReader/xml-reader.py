########
# parse voc xml file in detection
########


"""STEP"""
## read xml file or cache
## parse xml file (store its info and count objects)
## save as cache for next run


########
# format of xml file
########
"""
<annotation>
    <folder>***</folder>
    <filename>***</filename>
    <path>***</path>
    <size>
        <width>***</width>
        <height>***</height>
        <depth>***</depth>
    </size>
    <segmented>***</segmented>
    <object>
        <name>***</name>
        <bndbox>
            <xmin>***</xmin>
            <xmax>***</xmax>
            <ymin>***</ymin>
            <ymax>***</yamx>
        </bndbox>
    </object>
    <object>
        ******
    </object>
</annotation>
"""

########
# Code
########
import os
import xml.etree.ElementTree as ET
import pickle

def parse_voc_annotation(anno_dir, img_dir, cache_name, labels):
    ##########
    # format of cache
    """
    cache = {'all_insts": [
                            {
                            'filename': ***, 
                            'width': ***, 
                            'height': ***, 
                            'object': [
                                        {'name': ***, 'xmin': *, 'ymin': *, 'xmax': *, 'ymax': *}, 
                                        {'name': ***, 'xmin': *, 'ymin': *, 'xmax': *, 'yamx": *}, 
                                        ...
                                        ]
                            }, 
                            {...}, 
                            ...
                            ], 
                'seen_labels': {
                                <label>: <count>, 
                                ...
                                }
            }
    """
    ##########
    # format of tree from ET.parse()
    """
    tree --> leafs --> subleafs or attributions --> tag & text ?????
    """
    ##########
    if os.path.exists(cache_name):
        with open(cache_name) as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(anno_dir)):
            img = {'object': []}

            try:
                tree = ET.parse(os.path.join(anno_dir, ann))
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + os.path.join(anno_dir, ann))
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text # in case images have been removed to another dir
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = dim.text
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = dim.text
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = dim.text
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = dim.text
            # take the instance that has object in it
            if len(img['object']) > 0:
                all_insts += [img]

            cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
            with open(cache_name, 'wb') as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return all_insts, seen_labels