import argparse
from pathlib import Path
from datetime import datetime
import time
import os
import shutil
from tensorboardX import SummaryWriter
import torch
from torch.nn.utils import clip_grad_norm_
from models import ESCModel
from dataset import UrbanSound8KDataset


parser = argparse.ArgumentParser(description='ESC Fusion model')

parser.add_argument('mode', choices=['LMC', 'MC', 'MLMC', 'LMC+MC'])
parser.add_argument('--train_pickle', type=Path)
parser.add_argument('--test_pickle', type=Path)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
# parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
#                     metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

best_prec1 = 0
training_iterations = 0

experiment_name = '_'.join(('mode=' + args.mode,
                            'ep=' + str(args.epochs)))
experiment_dir = os.path.join(experiment_name, datetime.now().strftime('%b%d_%H-%M-%S'))
runs_path = Path('./runs')
if not runs_path.exists():
    runs_path.mkdir()
log_dir = runs_path / experiment_dir
summaryWriter = SummaryWriter(logdir=log_dir)


def main():
    global args, best_prec1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode != 'LMC+MC':
        print(args.mode)
        model = ESCModel()

    model = torch.nn.DataParallel(model, device_ids=None).to(device)

    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(args.train_pickle, args.mode),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(args.test_pickle, args.mode),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

    summaryWriter.close()


def train(train_loader, model, criterion, optimizer, epoch, device):
    global training_iterations

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        # compute output
        output = model(input)
        batch_size = input.size(0)

        target = target.to(device)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))

        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        training_iterations += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            summaryWriter.add_scalars('data/loss', {
                'training': losses.avg,
            }, training_iterations)
            summaryWriter.add_scalar('data/epochs', epoch, training_iterations)
            summaryWriter.add_scalar('data/learning_rate', optimizer.param_groups[-1]['lr'], training_iterations)
            summaryWriter.add_scalars('data/precision/top1', {
                'training': top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top5', {
                'training': top5.avg
            }, training_iterations)

            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.avg:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.avg:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.avg:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=optimizer.param_groups[-1]['lr']))

            print(message)


def validate(val_loader, model, criterion, device, name=''):
    global training_iterations

    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)

            # compute output
            output = model(input)
            batch_size = input.size(0)

            target = target.to(device)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        summaryWriter.add_scalars('data/loss', {
            'validation': losses.avg,
        }, training_iterations)
        summaryWriter.add_scalars('data/precision/top1', {
            'validation': top1.avg,
        }, training_iterations)
        summaryWriter.add_scalars('data/precision/top5', {
            'validation': top5.avg
        }, training_iterations)

        message = ('Testing Results: '
                   'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
                   'Loss {loss.avg:.5f}').format(top1=top1,
                                                 top5=top5,
                                                 loss=losses)
        print(message)

        return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    global experiment_dir
    weights_dir = Path('./models') / experiment_dir
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True)
    torch.save(state, weights_dir / filename)
    if is_best:
        shutil.copyfile(weights_dir / filename,
                        weights_dir / 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()