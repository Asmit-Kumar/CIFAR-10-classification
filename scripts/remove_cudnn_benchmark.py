import json
import glob

notebooks = [
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\05_resnet50_imagenet100.ipynb',
    r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\06_resnext50_imagenet100.ipynb'
]

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'torch.backends.cudnn.benchmark = False' in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
print("Removed torch.backends.cudnn.benchmark = False from the notebooks.")
