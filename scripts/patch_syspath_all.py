"""
Patch all notebooks: update sys.path.append('..') to sys.path.append('../..')
since notebooks moved from notebooks/ to notebooks/{dataset}/.
"""

import json
import glob

for nb_path in glob.glob('notebooks/**/*.ipynb', recursive=True):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        new_source = []
        for line in cell['source']:
            if "sys.path.append('..')" in line and "'../..'" not in line:
                line = line.replace("sys.path.append('..')", "sys.path.append('../..')")
                modified = True
            new_source.append(line)
        cell['source'] = new_source
    
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print(f'Patched: {nb_path}')
    else:
        print(f'Skipped: {nb_path}')
