#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to convert images to ktx textures

import os
import sys
import argparse
import subprocess
from datetime import datetime

def is_texture(fname):
    ext = os.path.splitext(fname)[1]
    return ext is not None and ext.lower() in (".tif", ".png", ".tga", ".jpg", ".jpeg")

def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE)
    try:
        r.check_returncode()
    except subprocess.CalledProcessError as e:
        print("[*] Error while running external command:\n")
        print("{}{}\n".format(r.stdout.decode('utf-8'),
                              r.stderr.decode('utf-8') if r.stderr else ""))
    return r.stdout.decode('utf-8')

def output_format_by_channels_textype(channels, textype):
    if channels in ('rgb', 'srgb'):
        if textype is not None and textype.lower() in ('n', 'normal'):
            return "GL_COMPRESSED_RG_RGTC2"
        else:
            return "GL_COMPRESSED_RGB_S3TC_DXT1_EXT"
    elif channels in ('rgba', 'srgba'):
        return "GL_COMPRESSED_RGBA_S3TC_DXT5_EXT"
    return "GL_COMPRESSED_RGB_S3TC_DXT1_EXT"

def process_texture(fullpath, out_fmt):
    flipped_path = fullpath + ".flip"
    out_path = fullpath + ".ktx"
    run_cmd(["convert", "-quiet", "-flip", fullpath, flipped_path])
    run_cmd(["any2ktx", "-i", out_fmt, flipped_path, out_path])
    os.remove(flipped_path)

def main():
    # Handle arguments
    args = parse_args()
    if args.dryrun:
        print("[+] Performing dry run!")

    search_dir = args.search_dir
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if is_texture(f):
                fullpath = os.path.join(root, f)
                fname    = os.path.splitext(f)[0]
                # Gather texture type and channels
                tex_type = fname[fname.rfind("_")+1:] if fname.rfind("_") != -1 else None
                channels = run_cmd(["identify", "-quiet", "-format", "%[channels]\n", fullpath]).splitlines()[0]
                # Print info
                print("[+] Processing ({}:{}) {}".format(channels, tex_type, fullpath))
                # Pick output format
                out_fmt  = output_format_by_channels_textype(channels, tex_type)
                # Process
                if not args.dryrun:
                    process_texture(fullpath, out_fmt)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert textures to orb compliant format")
    parser.add_argument("-n", "--dry-run",
                        action="store_true", dest="dryrun",
                        default=False, help="Dry run")
    parser.add_argument("search_dir", nargs="?",
                        default=".", help="Search directory")
    return parser.parse_args()

# Entrypoint
if __name__ == '__main__':
    time_point = datetime.now()
    main()
    ellapsed = datetime.now() - time_point
    print("[+] Total time: {} secs".format(ellapsed.total_seconds()))
