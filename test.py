# 必要なライブラリのインポート
import argparse
import os.path as osp
import cv2
import torch
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
import numpy as np

# コマンドライン引数の解析関数
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Image Processing')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--input', default="./input/dogs.jpg", help='path to input image')
    parser.add_argument('--output', default="./output/result_image.jpg", help='path to save output image')
    parser.add_argument('--score-thr', type=float, default=0.05, help='score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.7, help='NMS threshold')
    args = parser.parse_args()
    return args

# 描画関数
def draw_bounding_boxes(image, bboxes, scores, labels=None):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels and len(labels) > i:
            cv2.putText(image, labels[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        if scores is not None:
            score_text = f'{scores[i]:.2f}'
            cv2.putText(image, score_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    return image

# 画像処理関数
def process_image(runner, input_path, output_path, score_thr, nms_thr):
    # 画像を読み込み
    image = Image.open(input_path)
    image.save('temp.jpg')  # PIL画像を一時的に保存
    texts = [['yellow dog']]  # テキスト情報は使用しないため空のリストを設定
    data_info = dict(img_id=0, img_path='temp.jpg', texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0), data_samples=[data_info['data_samples']])

    # モデルで予測
    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    # NMSとスコア閾値によるフィルタリング
    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_bboxes = pred_instances.bboxes[keep].cpu().numpy()
    pred_scores = pred_instances.scores[keep].cpu().numpy()

    # 条件に基づくフィルタリング後のスコアとバウンディングボックスを取得
    filtered_indices = pred_scores > score_thr
    filtered_bboxes = pred_bboxes[filtered_indices]
    filtered_scores = pred_scores[filtered_indices]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGBからBGRへ変換
    image = draw_bounding_boxes(image, filtered_bboxes, filtered_scores)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGRからRGBへ変換
    cv2.imwrite(output_path, image)  # 結果を保存

if __name__ == '__main__':
    args = parse_args()

    # コンフィグをロード
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Runnerを構築
    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    # 画像を処理
    process_image(runner, args.input, args.output, args.score_thr, args.nms_thr)
