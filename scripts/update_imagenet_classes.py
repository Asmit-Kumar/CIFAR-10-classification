import json, os

nb_paths = [
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\07_convnext_tiny_imagenet100.ipynb',
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\06_resnext50_imagenet100.ipynb',
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\05_resnet50_imagenet100.ipynb',
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\04_resnet34_transfer_imagenet100.ipynb'
]

changed_files = []
for nb_path in nb_paths:
    if not os.path.exists(nb_path):
        continue
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'CIFAR100_CLASSES' in line:
                    line = line.replace('CIFAR100_CLASSES', 'IMAGENET100_CLASSES')
                    changed = True
                new_source.append(line)
            cell['source'] = new_source
            
    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        changed_files.append(nb_path)

print('Updated:', changed_files)
