"""
Patch all notebooks: update ../checkpoint/ to ../../checkpoint/ and ../logs/ to ../../logs/
since notebooks moved from notebooks/ to notebooks/{dataset}/.
"""

import json
import glob

replacements = {
    "'../checkpoint/": "'../../checkpoint/",
    "'../logs/": "'../../logs/",
}

for nb_path in glob.glob('notebooks/**/*.ipynb', recursive=True):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        new_source = []
        for line in cell['source']:
            original = line
            for old, new in replacements.items():
                line = line.replace(old, new)
            if line != original:
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
