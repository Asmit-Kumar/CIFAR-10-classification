import json
import os

source_nb = r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\05_resnext50_100.ipynb'
dest_nb = r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\06_resnext50_imagenet100.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        source = source.replace('CIFAR-100', 'ImageNet-100')
        cell['source'] = [line + '\n' for line in source.split('\n')]
    
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        new_source = []
        for line in source_lines:
            # Imports
            if 'from data.cifar_utils' in line:
                line = line.replace('get_cifar100_dataloaders', 'get_imagenet100_dataloaders')
                line = line.replace('get_cifar100_classes', 'get_imagenet100_classes')
                line = line.replace('data.cifar_utils', 'data.imagenet_utils')
                
            # Dataloaders
            if 'get_cifar100_dataloaders' in line:
                line = line.replace('get_cifar100_dataloaders', 'get_imagenet100_dataloaders')
                
            if 'batch_size=256' in line or 'batch_size=192' in line:
                line = 'batch_size = 128\n' # prevent OOM for ImageNet
                
            if 'trainloader, valloader, testloader =' in line and 'img_size' not in "".join(source_lines):
                # We need to add img_size=224 for imagenet
                pass # Will handle below with regex/replace
                
            # Classes
            if 'get_cifar100_classes()' in line:
                line = line.replace('get_cifar100_classes()', 'get_imagenet100_classes()')
                
            # NAME
            if "NAME = 'resnext50_100'" in line:
                line = "NAME = 'resnext50_100_imagenet'\n"
                
            # Fit method
            if 'train_losses, val_losses, val_accuracies = fit(' in "".join(source_lines) and 'step_scheduler_per_batch=True, log=True,' in line:
                 line = line.replace('log=True,', 'log=True, resume_from_checkpoint=True,')
                 
            new_source.append(line)
            
        # Post process dataloader call to add img_size=224 if it's imagenet
        source_str = "".join(new_source)
        if 'get_imagenet100_dataloaders(' in source_str and 'img_size=224' not in source_str:
            new_source_2 = []
            for line in new_source:
                if 'get_imagenet100_dataloaders(' in line:
                    if ')' in line:
                        line = line.replace(')', ', img_size=224)')
                    else:
                        new_source_2.append(line)
                        new_source_2.append('    img_size=224,\n')
                        continue
                new_source_2.append(line)
            new_source = new_source_2
            
        # Add cudnn benchmark=False in training cell
        if 'train_losses, val_losses, val_accuracies = fit(' in source_str:
            new_source.insert(0, "torch.backends.cudnn.benchmark = False # Disable to prevent VRAM spikes during training\n")
            
        cell['source'] = new_source
        
        # Clear outputs
        cell['outputs'] = []
        cell['execution_count'] = None

with open(dest_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created {dest_nb}")
