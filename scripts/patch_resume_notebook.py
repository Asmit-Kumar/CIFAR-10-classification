import json

nb_path = r'c:\Users\asmit\PycharmProjects\CIFAR_10\notebooks\05_resnet50_imagenet100.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Markdown cell ────────────────────────────────────────────────────────────
resume_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Resume Training from Checkpoint\n',
        '\n',
        'The run above crashed at epoch\u00a036 with `CUDNN_STATUS_EXECUTION_FAILED`.\n',
        'Root cause: `torch.backends.cudnn.benchmark\u00a0= True` lets cuDNN pick a faster\n',
        'convolution algorithm mid-run; around epoch\u00a036 it selected a workspace-heavy\n',
        'algorithm that pushed VRAM usage past the available\u00a016\u00a0GB, causing the sudden\n',
        'spike and crash.\n',
        '\n',
        'Fix: disable `benchmark`, clear the GPU cache, reload `_latest.pth`\n',
        '(epoch\u00a035 weights), and reconstruct `OneCycleLR` at the correct step so the\n',
        'LR schedule continues seamlessly from where it was interrupted.',
    ],
    'id': 'resume_md',
}

# ── Code cell source lines ───────────────────────────────────────────────────
src = []
src.append('import torchvision.models as models\n')
src.append('\n')
src.append('# 1. Clear VRAM; disable cuDNN autotuner (was True during crashed run)\n')
src.append('torch.cuda.empty_cache()\n')
src.append('torch.backends.cudnn.benchmark = False\n')
src.append('\n')
src.append('# 2. Rebuild dataloaders (reduce batch_size to 128 if OOM recurs)\n')
src.append('batch_size = 192\n')
src.append('trainloader, valloader, testloader = get_imagenet100_dataloaders(\n')
src.append('    batch_size=batch_size, num_workers=4, img_size=224,\n')
src.append(')\n')
src.append('\n')
src.append('# 3. Rebuild model and load the latest checkpoint (epoch 35)\n')
src.append('model = models.resnet50(weights=None)\n')
src.append('model.fc = nn.Linear(model.fc.in_features, 100)\n')
src.append('model.to(device)\n')
src.append('\n')
src.append("NAME = 'resnet50_100_imagenet'\n")
src.append("latest_ckpt = f'../checkpoint/{NAME}_latest.pth'\n")
src.append('model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=True))\n')
src.append("print(f'Loaded checkpoint: {latest_ckpt}')\n")
src.append('\n')
src.append('# 4. Recreate optimiser, scheduler and scaler\n')
src.append('epochs_total  = 90\n')
src.append('epochs_done   = 35          # completed before crash\n')
src.append('epochs_remain = epochs_total - epochs_done  # 55\n')
src.append('\n')
src.append('criterion = nn.CrossEntropyLoss()\n')
src.append('optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)\n')
src.append('\n')
src.append('steps_per_epoch = len(trainloader)               # 610\n')
src.append('steps_done      = epochs_done * steps_per_epoch  # 35 * 610 = 21350\n')
src.append('\n')
src.append("# last_epoch = steps_done - 1 so __init__()'s internal step() advances the\n")
src.append('# scheduler to step `steps_done`, matching the LR at the end of epoch 35.\n')
src.append('scheduler = optim.lr_scheduler.OneCycleLR(\n')
src.append('    optimizer,\n')
src.append('    max_lr=0.1,\n')
src.append('    steps_per_epoch=steps_per_epoch,\n')
src.append('    epochs=epochs_total,\n')
src.append("    pct_start=0.10,\n")
src.append("    anneal_strategy='cos',\n")
src.append('    div_factor=10,\n')
src.append('    final_div_factor=1e4,\n')
src.append('    last_epoch=steps_done - 1,  # resumes LR schedule from correct position\n')
src.append(')\n')
src.append("print(f'Scheduler LR at resume: {scheduler.get_last_lr()[0]:.6f}  (expected ~0.076677)')\n")
src.append('\n')
src.append("scaler = torch.amp.GradScaler('cuda')\n")
src.append('checkpoint = ModelCheckpoint(\n')
src.append('    model=model,\n')
src.append("    checkpoint_path=f'../checkpoint/{NAME}_latest.pth',\n")
src.append("    best_model_path=f'../checkpoint/{NAME}_best.pth',\n")
src.append("    mode='max',\n")
src.append('    verbose=True,\n')
src.append(')\n')
src.append('\n')
src.append('# 5. Continue training for the remaining 55 epochs\n')
src.append('train_losses, val_losses, val_accuracies = fit(\n')
src.append('    model, trainloader, valloader, criterion,\n')
src.append('    optimizer, scheduler, scaler, device, epochs_remain, checkpoint,\n')
src.append('    step_scheduler_per_batch=True, log=True,\n')
src.append(')')

resume_code = {
    'cell_type': 'code',
    'metadata': {},
    'source': src,
    'id': 'resume_code',
    'outputs': [],
    'execution_count': None,
}

# Insert the two new cells after index 15 (crashed training loop, id=ed6e74ba)
nb['cells'].insert(16, resume_code)
nb['cells'].insert(16, resume_md)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Done. Total cells: {len(nb["cells"])}')
ids = [(i, c.get("id", ""), c.get("cell_type", "")) for i, c in enumerate(nb["cells"])]
for x in ids:
    print(x)
