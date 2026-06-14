import json

nb_path = r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\05_resnet50_imagenet100.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Remove the resume cells we added earlier
# We look for cell ids "resume_md" and "resume_code"
nb['cells'] = [c for c in nb['cells'] if c.get('id') not in ('resume_md', 'resume_code')]

# 2. Find the main training cell
# The main training cell usually has "train_losses, val_losses, val_accuracies = fit("
for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = "".join(c['source'])
        if 'fit(' in source and 'trainloader' in source:
            # We want to add resume_from_checkpoint=True
            # And add torch.backends.cudnn.benchmark = False at the top
            
            # Reconstruct the source
            new_source = []
            added_cudnn = False
            for line in c['source']:
                if 'import' in line and not added_cudnn:
                    new_source.append(line)
                    new_source.append("import torch\n")
                    new_source.append("torch.backends.cudnn.benchmark = False # Disable to prevent VRAM spikes during training\n")
                    added_cudnn = True
                    continue
                
                if 'step_scheduler_per_batch=True, log=True,' in line:
                    line = line.replace('log=True,', 'log=True, resume_from_checkpoint=True,')
                new_source.append(line)
            
            # If we didn't add cudnn (no imports in cell), just prepend it
            if not added_cudnn:
                new_source.insert(0, "torch.backends.cudnn.benchmark = False # Disable to prevent VRAM spikes during training\n")
                
            c['source'] = new_source
            print("Updated fit() cell.")
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")
