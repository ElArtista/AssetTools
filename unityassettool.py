#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Helper script to convert unity prefab/scene formats to arbitary json format
"""

import os
import sys
import argparse
import yaml
import json

# Provide a simple pass through constructor for base nodes that are tagged with
# Unity's '!u!xxx' directive
def unity_yaml_constructor(loader, tag_suffix, node):
    value = loader.construct_mapping(node)
    return value
yaml.add_multi_constructor("tag:unity3d.com,2011:", unity_yaml_constructor)
# HACK: Unity yaml files miss the tag declaration between streams.
# Add it to the Loader's default (from language spec) list so it always resolves
yaml.Loader.DEFAULT_TAGS[u'!u!'] = "tag:unity3d.com,2011:"

# HACK: Unity yaml files misuse anchor functionality. Each unity yaml document
# is marked with an anchor that is never dereferrenced according to the yaml specification
# but is actually used by some "fileID" attributes in various objects.
# We wrap some parser functions to populate an anchor list in order to have access to this
# data that are by default discarded/processed by the yaml parser.
yaml_default_load_all_fn = yaml.load_all
yaml_default_get_event_fn = yaml.Loader.get_event
anchor_list = []
def yaml_loader_get_event_wrapper(self):
    global anchor_list
    ev = yaml_default_get_event_fn(self)
    if isinstance(ev, yaml.CollectionStartEvent) and ev.anchor is not None:
        anchor_list.append(int(ev.anchor))
    return ev
def yaml_load_all_with_document_anchors(stream, Loader=yaml.Loader):
    global anchor_list
    anchor_list = []
    return (yaml_default_load_all_fn(stream, Loader), anchor_list)
yaml.load_all = yaml_load_all_with_document_anchors
yaml.Loader.get_event = yaml_loader_get_event_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_dir", help="path to unity project folder")
    return parser.parse_args()

def guid_from_meta_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            metadict = yaml.load(yamlfdata)
            guid = metadict["guid"]
            return guid
    return None

def add_missing_unity_yaml_alias(yamlfdata):
    pass

def mat_from_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            metadict = yaml.load(yamlfdata)
            tex_nodes = metadict["Material"]["m_SavedProperties"]["m_TexEnvs"]
            mat = {}
            for tn in tex_nodes:
                if tn["second"]["m_Texture"]["fileID"] != 0:
                    tex_type = tn["first"]["name"][1:]
                    tex_guid = tn["second"]["m_Texture"]["guid"]
                    mat[tex_type] = tex_guid
            return mat
    return None

def prefab_from_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            yparsed_data, anchors = yaml.load_all(yamlfdata)
            metadicts = list(yparsed_data)
            # TODO!! Support prefabs with multiple meshes
            if "MeshRenderer" in metadicts[2]:
                mat_guids = []
                for mn in metadicts[2]["MeshRenderer"]["m_Materials"]:
                    mat_guids.append(mn["guid"])
                mdl_guid = metadicts[3]["MeshFilter"]["m_Mesh"]["guid"]
                prefab = {}
                prefab["materials"] = mat_guids
                prefab["model"] = mdl_guid
                return prefab
    return None

def scene_from_file(f):
    with open(f, "r") as mf:
        yamlfdata = mf.read()
        if yamlfdata:
            yparsed_data, anchors = yaml.load_all(yamlfdata)
            scene = {}
            scene["objects"] = {}
            for idx, md in enumerate(yparsed_data):
                if "GameObject" in md:
                    go = md["GameObject"]
                    obj_name = go["m_Name"]
                    scene["objects"][obj_name] = { "id": anchor_list[idx] }
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

def scan_materials(asset_dir, guididx, report_fn):
    process_fn = lambda f: mat_from_file(f)
    return scan_assets(asset_dir, guididx, [".mat"], report_fn, process_fn)

def scan_textures(asset_dir, guididx, report_fn):
    process_fn = lambda f: f
    ext_list = [".png", ".jpeg", ".jpg", ".tif", ".tga", ".bmp"]
    return scan_assets(asset_dir, guididx, ext_list, report_fn, process_fn)

def scan_models(asset_dir, guididx, report_fn):
    process_fn = lambda f: f
    return scan_assets(asset_dir, guididx, [".fbx"], report_fn, process_fn)

def scan_prefabs(asset_dir, guididx, report_fn):
    process_fn = lambda f: prefab_from_file(f)
    return scan_assets(asset_dir, guididx, [".prefab"], report_fn, process_fn)

def scan_scenes(asset_dir, guididx, report_fn):
    process_fn = lambda f: scene_from_file(f)
    return scan_assets(asset_dir, guididx, [".unity"], report_fn, process_fn)

#-----------------------------------------------------------------
def construct_json_output(assetmap):
    data = {}
    data["materials"] = assetmap["material"]
    data["textures"] = assetmap["texture"]
    data["models"] = assetmap["model"]
    #data["prefabs"] = assetmap["prefab"]
    data["scenes"] = assetmap["scene"]
    return json.dumps(data, indent=4)

def main():
    args = parse_args()
    asset_dir = os.path.join(args.proj_dir, "Assets")
    if not os.path.exists(asset_dir):
        print("Error: could not locate \"Assets\" directory, are you using a valid unity project path?")
        return

    # Scan .meta files and construct map with guids to filepaths
    clear_line = 128 * " " + "\r"
    report_fn = lambda f: print(clear_line + "[+] Parsing metadata file: %s" % (f), end='\r')
    guidmap = scan_meta_files(asset_dir, report_fn)
    print(clear_line + "[+] Parsing metadata files done.")
    # Create inverse map
    inv_map = {v: k for k, v in guidmap.items()}

    assettypes = ['material', 'texture', 'model', 'prefab', 'scene']
    assetscanners = [scan_materials, scan_textures, scan_models, scan_prefabs, scan_scenes]
    assetmap = {}
    for at in assettypes:
        report_fn = lambda f: print(clear_line + "[+] Processing %s file: %s" % (at, f), end='\r')
        idx = assettypes.index(at)
        assetmap[at] = assetscanners[idx](asset_dir, inv_map, report_fn)
        print(clear_line + "[+] Processing %s files done." % (at))

    print("[+] Generating json output...")
    json_out = construct_json_output(assetmap)
    print(json_out)

if __name__ == '__main__':
    main()
