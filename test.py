import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

from dataset import UrbanSound8KDataset
from models import ESCModel
import numpy as np
import pickle


def print_accuracy(scores, labels, fname, mapping, average_segments=False):
    if average_segments:
        unique_fname = np.unique(fname)
        pred = np.zeros(unique_fname.shape[0])
        gt = np.zeros(unique_fname.shape[0])
        for i, uid in enumerate(unique_fname):
            idx = fname == uid
            idx = idx.squeeze()
            untrimmed_scores = scores[idx]
            untrimmed_label = labels[idx][0]
            avg_untrimmed_scores = np.mean(untrimmed_scores, axis=0)
            pred[i] = np.argmax(avg_untrimmed_scores)
            gt[i] = untrimmed_label
    else:
        pred = [np.argmax(score) for score in scores]
        gt = labels

    cf = confusion_matrix(gt, pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_cnt[cls_hit == 0] = 1  # to avoid divisions by zero
    cls_acc = cls_hit / cls_cnt

    acc = accuracy_score(gt, pred)

    print('Accuracy {:.02f}%'.format(acc * 100))
    print('Per-class accuracies:')
    for i in range(len(cls_acc)):
        print('{}: {:.02f}%'.format(mapping[i], cls_acc[i] * 100))
    print('Average Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


def main():

    parser = argparse.ArgumentParser(description="Standard video-level" +
                                     " testing")
    parser.add_argument('mode', choices=['LMC', 'MC', 'MLMC', 'LMC+MC'])
    parser.add_argument('weights_dir', type=str)
    parser.add_argument('--scores_input', nargs='+', type=Path)
    parser.add_argument('--scores_output', type=Path)
    parser.add_argument('--test_pickle')
    parser.add_argument('--mapping')
    parser.add_argument('--scores_root', type=str, default='scores')
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--average', action='store_true')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode != 'LMC+MC':
        print(args.mode)
        net = ESCModel()

        weights = '{weights_dir}/model_best.pth.tar'.format(
            weights_dir=args.weights_dir)
        checkpoint = torch.load(weights)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        net.load_state_dict(base_dict)

        test_loader = torch.utils.data.DataLoader(
            UrbanSound8KDataset(args.test_pickle, args.mode),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        net = torch.nn.DataParallel(net, device_ids=None).to(device)
        with torch.no_grad():
            net.eval()

            results = []
            total_num = len(test_loader.dataset)

            proc_start_time = time.time()
            max_num = args.max_num if args.max_num > 0 else total_num
            for i, (data, label, fname) in enumerate(test_loader):
                if i >= max_num:
                    break
                data = data.to(device)
                rst = net(data)
                rst = rst.cpu().numpy().squeeze()
                label_ = label.item()
                results.append((rst, label_, fname))

                cnt_time = time.time() - proc_start_time
                print('video {} done, total {}/{}, average {} sec/video'.format(
                    i, i + 1, total_num, float(cnt_time) / (i + 1)))

        if not args.scores_output.parent.exists():
            args.scores_output.parent.mkdir(parents=True)
        pickle.dump(results, open(args.scores_output, 'wb'))
        scores = np.array([res[0] for res in results])
        labels = np.array([res[1] for res in results])
        fname = np.array([res[2] for res in results])
    else:
        lmc_results = pickle.load(open(args.scores_input[0], 'rb'))
        mc_results = pickle.load(open(args.scores_input[1], 'rb'))
        lmc_scores = np.array([res[0] for res in lmc_results])
        mc_scores = np.array([res[0] for res in mc_results])
        print(np.array([lmc_scores, mc_scores]).shape)
        scores = np.mean(np.array([lmc_scores, mc_scores], axis=0))
        labels = np.array([res[1] for res in lmc_results])
        fname = np.array([res[2] for res in lmc_results])

    print_accuracy(scores,
                   labels,
                   fname,
                   pickle.load(open(args.mapping, 'rb')),
                   average_segments=args.average)


if __name__ == '__main__':
    main()
