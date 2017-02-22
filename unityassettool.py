#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Helper script to convert unity prefab/scene formats to arbitary json format
"""

from __future__ import print_function
import os
import re
import sys
import argparse
import yaml
import json
import time

# Use more performant libyaml backend
LOADER = yaml.CLoader

# Provide a simple pass through constructor for base nodes that are tagged with
# Unity's '!u!xxx' directive
def unity_yaml_constructor(loader, tag_suffix, node):
    value = loader.construct_mapping(node)
    return value
yaml.add_multi_constructor("tag:unity3d.com,2011:", unity_yaml_constructor, Loader=LOADER)

# HACK: Unity yaml files miss the tag declaration between streams.
# Add it to the Loader's default (from language spec) list so it always resolves
yaml.Loader.DEFAULT_TAGS[u'!u!'] = "tag:unity3d.com,2011:"

# HACK: Unity yaml files miss the tag declaration between streams.
# Preprocess yaml source to add the tag declaration in every stream
def yaml_preproc_stream(stream):
    # Remove '%TAG !u! tag:unity3d.com,2011:' lines
    # and prepend '%TAG !u! tag:unity3d.com,2011:' line before every '--- !u!xxx' line
    stream = stream.replace('\n%TAG !u! tag:unity3d.com,2011:\n','\n') \
                   .replace('\n--- !u!', '\n%TAG !u! tag:unity3d.com,2011:\n--- !u!')
    return stream

# HACK: Unity yaml files misuse anchor functionality. Each unity yaml document
# is marked with an anchor that is never dereferrenced according to the yaml specification
# but is actually used by some "fileID" attributes in various objects.
# We wrap some parser functions to populate an anchor list in order to have access to this
# data that are by default discarded/processed by the yaml parser.

# Global cleared on every load_all invocation
anchor_list = []

# With yaml.Loader use a get_event wrapper to gather anchors
yaml_default_get_event_fn = LOADER.get_event
def yaml_loader_get_event_wrapper(self):
    global anchor_list
    ev = yaml_default_get_event_fn(self)
    if isinstance(ev, yaml.CollectionStartEvent) and ev.anchor is not None:
        anchor_list.append(int(ev.anchor))
    return ev
LOADER.get_event = yaml_loader_get_event_wrapper

# With yaml.CLoader scan stream for document anchors
def yaml_gather_doc_anchors(stream):
    anchors = re.findall(r'^--- !u!\d+\s+&(\d+)', stream, flags=re.MULTILINE)
    return [int(a) for a in anchors]

# Wrap load_all function to return per document anchor
yaml_default_load_all_fn = yaml.load_all
def yaml_load_all_with_document_anchors(stream, Loader=LOADER):
    global anchor_list
    anchor_list = []
    if Loader == yaml.CLoader:
        stream = yaml_preproc_stream(stream)
        anchor_list = yaml_gather_doc_anchors(stream)
    return (yaml_default_load_all_fn(stream, Loader), anchor_list)
yaml.load_all = yaml_load_all_with_document_anchors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("search_dir", help="folder to recursively search for assets")
    return parser.parse_args()

def guid_from_meta_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            metadict = yaml.load(yamlfdata, Loader=yaml.CLoader)
            guid = metadict["guid"]
            return guid
    return None

def mat_from_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            metadicts, anchors = yaml.load_all(yamlfdata)
            tex_nodes = list(metadicts)[0]["Material"]["m_SavedProperties"]["m_TexEnvs"]
            mat = {}
            for tn in tex_nodes:
                if tn["second"]["m_Texture"]["fileID"] != 0:
                    tex_type = tn["first"]["name"][1:]
                    tex_guid = tn["second"]["m_Texture"]["guid"]
                    tex_scal = tn["second"]["m_Scale"]
                    mat[tex_type] = {"texture": tex_guid, "scale": [tex_scal["x"], tex_scal["y"]]}
            return mat
    return None

def game_object_from_prefab(obj, prfb):
    if "model" in prfb:
        obj["model"] = prfb["model"]
    if "transform" in prfb:
        obj["transform"] = prfb["transform"]
    if "materials" in prfb:
        obj["materials"] = prfb["materials"].copy()

def game_objects_from_docs(docs, prefabs=None):
    sobjs = {}
    # Populate full game object list first with prefab references
    for doc_id, md in docs.items():
        if "GameObject" in md:
            go = md["GameObject"]
            obj = {}
            obj["name"] = go["m_Name"]
            # Populate from prefab
            if go["m_PrefabParentObject"]["fileID"] != 0 and prefabs:
                prfb_guid = go["m_PrefabParentObject"]["guid"]
                prfb_fid  = go["m_PrefabParentObject"]["fileID"]
                if prfb_guid in prefabs and prfb_fid in prefabs[prfb_guid]:
                    prfb = prefabs[prfb_guid]
                    game_object_from_prefab(obj, prfb[prfb_fid])
                    obj["prefab_guid"] = prfb_guid
            # Use document anchor as object id
            sobjs[doc_id] = obj
    # Populate full game objects with data
    for goid, go in sobjs.items():
        for c in docs[goid]["GameObject"]["m_Component"]:
            comp_id = c["component"]["fileID"]
            component = docs[comp_id]
            if "MeshRenderer" in component:
                mat_guids = []
                for mn in component["MeshRenderer"]["m_Materials"]:
                    mat_guids.append(mn["guid"])
                if not prefabs or (prefabs and "prefab_guid" not in go):
                    go["materials"] = mat_guids
            elif "MeshFilter" in component:
                mm = component["MeshFilter"]["m_Mesh"]
                mdl_guid = mm["guid"]
                go["model"] = mdl_guid
            elif "Transform" in component or "RectTransform" in component:
                component = next(iter(component.values()))
                goid = component["m_GameObject"]["fileID"]
                pc = component["m_LocalPosition"]
                rc = component["m_LocalRotation"]
                sc = component["m_LocalScale"]
                # NOTE: Converting handness
                trans = {
                    "position" : [-pc["x"], pc["y"], pc["z"]],
                    "rotation" : [rc["x"], -rc["y"], -rc["z"], rc["w"]],
                    "scale"    : [sc["x"], sc["y"], sc["z"]]
                }
                go["transform"] = trans
                parnt_trans = component["m_Father"]["fileID"]
                # Populate parent link
                if parnt_trans != 0:
                    parnt = next(iter(docs[parnt_trans].values()))["m_GameObject"]["fileID"]
                    go["transform"]["parent"] = str(parnt)
                # Populate child link
                child_transfs = [str(next(iter(docs[e["fileID"]].values()))["m_GameObject"]["fileID"]) for e in component["m_Children"]]
                if child_transfs:
                    go["transform"]["children"] = child_transfs
    for sobj_id, sobj in sobjs.copy().items():
        if "children" not in sobj["transform"] and "model" not in sobj:
            if "parent" in sobj["transform"]:
                prnt = sobjs[int(sobj["transform"]["parent"])]
                prnt["transform"]["children"].remove(str(sobj_id))
            del sobjs[sobj_id]
    return sobjs

def merge_same_model_gameobjects(docs, gobjs):
    merged_gobjs = {}
    root_obj_ids = [obj_id for obj_id, obj in gobjs.items() if "parent" not in obj["transform"]]
    queue = []
    queue.extend(root_obj_ids)
    obj_id = None
    while len(queue):
        # Store last element
        last_elem = obj_id
        # Get current element
        obj_id = queue.pop(0)
        # Check if element is mergable
        obj = gobjs[obj_id]
        if "children" not in obj["transform"]:
            merged = False
            if "model" in obj:
                if "parent" in obj["transform"]:
                    # Check if mergable with parent
                    prnt_id = int(obj["transform"]["parent"])
                    prnt_obj = merged_gobjs[prnt_id]
                    if "model" in prnt_obj and prnt_obj["model"] == obj["model"]:
                        prnt_obj["materials"].extend(obj["materials"])
                        obj_id = prnt_id
                        merged = True
                    # Check if mergable with previous sibling
                    if not merged and last_elem:
                        sibl_id = last_elem
                        sibl_obj = merged_gobjs[sibl_id]
                        # Check for same parent, thus siblings
                        if "parent" in sibl_obj["transform"] and sibl_obj["transform"]["parent"] == obj["transform"]["parent"]:
                            if "model" in sibl_obj and sibl_obj["model"] == obj["model"]:
                                if "materials" in obj:
                                    if "materials" not in sibl_obj:
                                        sibl_obj["materials"] = []
                                    sibl_obj["materials"].extend(obj["materials"])
                                # Last element must remain the same now that we merged
                                obj_id = sibl_id
                                merged = True
            if not merged:
                merged_gobjs[obj_id] = obj
        else:
            # Add its children to be processed
            chld_ids = [int(chld_id) for chld_id in obj["transform"]["children"]]
            # Sort by prefab fileID
            sorted_chld_ids = sorted(chld_ids, key=lambda x: docs[x]["GameObject"]["m_PrefabParentObject"]["fileID"])
            queue.extend(sorted_chld_ids)
            # Add the element to the processed list
            merged_gobjs[obj_id] = obj
    return merged_gobjs

def prefab_from_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            yparsed_data, anchors = yaml.load_all(yamlfdata)
            metadicts = list(yparsed_data)
            # Get Prefab document
            pf = next((md["Prefab"] for md in metadicts if "Prefab" in md), None)
            if pf is not None:
                docs = {anchors[i]: metadicts[i] for i in range(len(anchors))}
                prefab_gobjs = game_objects_from_docs(docs)
                return prefab_gobjs
    return None

def scene_from_file(f, prefabs):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            yparsed_data, anchors = yaml.load_all(yamlfdata)
            scene = {}
            metadicts = list(yparsed_data)
            # Construct a more useful dictionary from fileID to document
            docs = {anchors[i]: metadicts[i] for i in range(len(anchors))}
            gobjs = game_objects_from_docs(docs, prefabs)
            merged_objs = merge_same_model_gameobjects(docs, gobjs)
            #eprint("UNMERGED: " + str(len(gobjs)) + " MERGED: " + str(len(merged_objs)))
            scene["objects"] = merged_objs
            return scene
    return None

def scan_meta_files(asset_dir, report_fn):
    """ Constructs a map of guids and their corresponding filepaths """
    guidmap = {}
    for root, dirs, files in os.walk(asset_dir):
        for f in files:
            fname, fext = os.path.splitext(f)
            if fext == ".meta":
                report_fn(f)
                guid = guid_from_meta_file(os.path.join(root, f))
                guidmap[guid] = os.path.join(root, fname)
    return guidmap

def scan_assets(asset_dir, guididx, ext_list, report_fn, asset_process_fn):
    """
    Constructs a map of guids and their corresponding asset data
        asset_process_fn: function that takes relative path to the asset file
            and returns object to be stored as a value to the map (usually is an identity function)
    """
    assetmap = {}
    for root, dirs, files in os.walk(asset_dir):
        for f in files:
            fname, fext = os.path.splitext(f)
            if fext.lower() in ext_list:
                report_fn(f)
                relpath = os.path.join(root, f)
                asset = asset_process_fn(relpath)
                guid = guididx[relpath]
                assetmap[guid] = asset
    return assetmap

def scan_materials(asset_dir, guididx, report_fn, proc_args):
    process_fn = lambda f: mat_from_file(f)
    return scan_assets(asset_dir, guididx, [".mat"], report_fn, process_fn)

def scan_textures(asset_dir, guididx, report_fn, proc_args):
    process_fn = lambda f: f
    ext_list = [".png", ".jpeg", ".jpg", ".tif", ".tga", ".bmp"]
    return scan_assets(asset_dir, guididx, ext_list, report_fn, process_fn)

def scan_models(asset_dir, guididx, report_fn, proc_args):
    process_fn = lambda f: f
    return scan_assets(asset_dir, guididx, [".fbx"], report_fn, process_fn)

def scan_prefabs(asset_dir, guididx, report_fn, proc_args):
    process_fn = lambda f: prefab_from_file(f)
    return scan_assets(asset_dir, guididx, [".prefab"], report_fn, process_fn)

def scan_scenes(asset_dir, guididx, report_fn, proc_args):
    prefabs = proc_args['prefab']
    process_fn = lambda f: scene_from_file(f, prefabs)
    return scan_assets(asset_dir, guididx, [".unity"], report_fn, process_fn)

#-----------------------------------------------------------------
def tex_is_used(tex_ref, data):
    for mat in data["materials"].values():
        for mat_type in mat.values():
            if tex_ref == mat_type["texture"]:
                return True
    return False

def mdl_is_used(mdl_ref, data):
    for prfb in data["prefabs"].values():
        for prfb_obj in prfb.values():
            if "model" in prfb_obj.keys():
                if mdl_ref == prfb_obj["model"]:
                    return True
    for scene in data["scenes"].values():
        for scn_obj in scene["objects"].values():
            if "model" in scn_obj.keys():
                if mdl_ref == scn_obj["model"]:
                    return True
    return False

def data_cleanup(data):
    # Remove uneccessary models and textures
    for tex in data["textures"].copy().keys():
        if not tex_is_used(tex, data):
            del data["textures"][tex]
    for mdl in data["models"].copy().keys():
        if not mdl_is_used(mdl, data):
            del data["models"][mdl]
    # Remove invalid model and material references
    for scene in data["scenes"].values():
        for scn_obj in scene["objects"].values():
            if "model" in scn_obj.keys():
                if scn_obj["model"] not in data["models"]:
                    del scn_obj["model"]
                if "materials" in scn_obj.keys():
                    obj_mats = scn_obj["materials"]
                    for mat in obj_mats:
                        if mat not in data["materials"]:
                            obj_mats.remove(mat)

def construct_json_output(assetmap):
    data = {}
    data["materials"] = assetmap["material"]
    data["textures"] = assetmap["texture"]
    data["models"] = assetmap["model"]
    data["prefabs"] = assetmap["prefab"]
    data["scenes"] = assetmap["scene"]
    data_cleanup(data)
    return json.dumps(data, indent=4, sort_keys=True)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    # Profiling start
    start_time = time.time()

    # Search directory
    args = parse_args()
    search_dir = args.search_dir

    # Scan .meta files and construct map with guids to filepaths
    clear_line = 80 * " " + "\r"
    report_fn = lambda f: eprint(clear_line + "[+] Parsing metadata file: %s" % (f), end='\r')
    guidmap = scan_meta_files(search_dir, report_fn)
    eprint(clear_line + "[+] Parsing metadata files done.")
    # Create inverse map
    inv_map = {v: k for k, v in guidmap.items()}

    assetmap = {}
    assettypes = ['material', 'texture', 'model', 'prefab', 'scene']
    assetscanners = [scan_materials, scan_textures, scan_models, scan_prefabs, scan_scenes]
    assetscanargs = [None, None, None, None, assetmap]
    for at in assettypes:
        report_fn = lambda f: eprint(clear_line + "[+] Processing %s file: %s" % (at, f), end='\r')
        idx = assettypes.index(at)
        assetmap[at] = assetscanners[idx](search_dir, inv_map, report_fn, proc_args=assetscanargs[idx])
        eprint(clear_line + "[+] Processing %s files done." % (at))

    eprint("[+] Generating json output...")
    json_out = construct_json_output(assetmap)
    print(json_out)

    # Show execution time
    tot_time = time.time() - start_time
    eprint("Total time: %f" %(tot_time))

if __name__ == '__main__':
    main()
