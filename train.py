import argparse
from detr import build_model
from utils.config import load_config, Config
from tinygrad.tinygrad import Tensor
import tinygrad.tinygrad.nn.optim as optim
import datetime
import time
import sys
from logger import DetrLogger
import math

def train(config: Config):
    LOGGER = DetrLogger()
    Tensor.manual_seed(config.runtime.seed)
    model, criterion = build_model(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.training.lr_backbone,
        },
    ]

    optimizer = optim.AdamW(param_dicts, lr=config.training.lr, weight_decay=config.training.weight_decay)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # TODO: Create datasets
    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # TODO: Build dataloaders
    # ----------------------------------------
    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)
    # ----------------------------------------

    # TODO: Load from checkpoint?
    # ----------------------------------------
    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # output_dir = Path(args.output_dir)
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1
    # ----------------------------------------

    # TODO: Eval in another script?
    # ----------------------------------------
    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return
    # ----------------------------------------

    print("Start training")
    for epoch in range(config.training.epochs):
        model.train()
        criterion.train()

        for samples, targets in data_loader_train:
            targets = [{k: v for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict)
        
        if not math.isfinite(losses): LOGGER.info_no_save("Loss is inf, stopping..."); sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        LOGGER.info("Epoch__lr", epoch, optimizer.param_groups[0]["lr"])
        LOGGER.info("Epoch__TrainLoss_ClassError", epoch, loss_dict['class_error'])
        LOGGER.info("Epoch__TrainLoss", epoch, losses)

        # train_stats = train_one_epoch(
        #     model, criterion, data_loader_train, optimizer, device, epoch,
        #     args.clip_max_norm)
        # lr_scheduler.step()
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     # extra checkpoint before LR drop and every 100 epochs
        #     if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
        #         checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        #     # for evaluation logs
        #     if coco_evaluator is not None:
        #         (output_dir / 'eval').mkdir(exist_ok=True)
        #         if "bbox" in coco_evaluator.coco_eval:
        #             filenames = ['latest.pth']
        #             if epoch % 50 == 0:
        #                 filenames.append(f'{epoch:03}.pth')
        #             for name in filenames:
        #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                            output_dir / "eval" / name)

    seconds_since_start = str(datetime.timedelta(seconds=int(time.time() - LOGGER.start_time)))
    print('It took {} seconds to train'.format(seconds_since_start))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', add_help=False)
    parser.add_argument('--config', default='train.yaml', type=str, help='path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
