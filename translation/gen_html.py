"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Usage:
# cd day2golden_results
# python gen_html.py -i ./results
"""

import argparse
import os
import glob

def append_index(filesets, style_paths, input_dir):
    index_path = os.path.join(input_dir, "index.html")

    index = open(index_path, "w")
    index.write("<html><body>")
    index.write("<h1></h1>")
    index.write("<table><tr>")
    index.write("<th>source / style</th>")

    for path in style_paths:
        index.write("<td><img src='0/%s'  width='256'></td>" % os.path.basename(path))
    index.write("</tr>")

    for fileset in filesets:   
        if os.path.isdir(fileset):
            file_name = os.path.basename(fileset)
            index.write("<tr>")
            input_path = glob.glob(os.path.join(fileset, "input.jpg"))
            output_paths = sorted(glob.glob(os.path.join(fileset, "output_*_blend.*")))
            # output_paths = sorted(glob.glob(os.path.join(fileset, "output_*_opt.jpg"))) #optional show optimized results

            index.write("<td><img src='%s/input.jpg' width='256'></td>" % file_name)

            for path in output_paths:
                index.write("<td><img src='%s/%s'  width='256'></td>" % (file_name, os.path.basename(path)))

            index.write("</tr>")
    return index_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="./", help="path to folder containing images")
    a = parser.parse_args()

    input_folders = sorted(glob.glob(os.path.join(a.input_dir, "*")))
    print("%d folders to be processed."%(len(input_folders)))

    style_folder = input_folders[0]
    style_paths = sorted(glob.glob(os.path.join(style_folder, "style_*.jpg")))

    print("%d style images"%(len(style_paths)))
    append_index(input_folders, style_paths, a.input_dir)





