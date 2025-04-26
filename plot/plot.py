
import os
import json
import math
import argparse
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_metrics(sample_data):
    # 找出所有不以 "_T" 結尾，且有對應 "_T" 的 key
    return [k for k in sample_data.keys() if not k.endswith('_T') and f'{k}_T' in sample_data]

def main():
    parser = argparse.ArgumentParser(
        description="從一或多個 JSON 檔裡的 `xxx`＆`xxx_T` 自動畫出折線圖"
    )
    parser.add_argument('json_files', nargs='+',
                        help='要畫的 JSON 檔（可以一次丟多個）')
    parser.add_argument('-m', '--metrics', nargs='*',
                        help='要畫哪些指標，預設自動偵測')
    parser.add_argument('-o', '--out', default=None,
                        help='指定輸出圖檔名稱（如不設定則直接彈窗顯示）')
    args = parser.parse_args()

    # 載入所有資料
    all_data = {path: load_json(path) for path in args.json_files}

    # 自動或手動選指標
    if args.metrics:
        metrics = args.metrics
    else:
        # 取第一個檔案偵測
        sample = next(iter(all_data.values()))
        metrics = detect_metrics(sample)

    n = len(metrics)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    axes = axes.flatten()

    # 畫每個檔案、每個指標
    for path, data in all_data.items():
        label = os.path.splitext(os.path.basename(path))[0]
        for idx, metric in enumerate(metrics):
            x = data.get(f'{metric}_T', [])
            y = data.get(metric, [])
            if x and y and len(x)==len(y):
                axes[idx].plot(x, y, label=label)

    # 設定每張子圖
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_title(metric)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    # 若圖格多出空白就移除
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f'已儲存到 {args.out}')
    else:
        plt.show()

if __name__ == '__main__':
    main()
