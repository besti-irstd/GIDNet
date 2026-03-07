import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 定义数据集名称 (对应 datasets 文件夹下的名字)
# 注意：键(Key)是显示的名称，值(Value)是文件夹实际名称
# 特别注意：预测文件夹里的名字是大写 '1K'，这里我们做一个映射处理
DATASETS_CONFIG = [
    {
        "name": "IRSTD-1k",  # 显示在图表标题上的名字
        "gt_folder_name": "IRSTD-1k",  # datasets 里的文件夹名
        "pred_folder_name": "IRSTD-1K"  # predict/模型 里的文件夹名 (注意大小写差异)
    },
    {
        "name": "NUAA-SIRST",
        "gt_folder_name": "NUAA-SIRST",
        "pred_folder_name": "NUAA-SIRST"
    },
    {
        "name": "NUDT-SIRST",
        "gt_folder_name": "NUDT-SIRST",
        "pred_folder_name": "NUDT-SIRST"
    }
]

# 2. 定义9个模型名称 (严格对应 predict 文件夹下的名字)
MODELS = [
    "GIDNet(ours)",
    "HDNet",
    "L2SKNet-FPN",
    "L2SKNet-UNet",
    "MMLNet",
    "MSHNet",
    "SDS-Net",
    "UIU-Net"
]

# 3. 定义根目录路径 (请根据您实际运行代码的位置修改这里)
ROOT_DIR = "D:\\MCDFNet\\MCDFNet"  # 假设代码在截图显示的根目录下运行
GT_ROOT = os.path.join(ROOT_DIR, "datasets")
PRED_ROOT = os.path.join(ROOT_DIR, "predict")

# 4. 定义绘图颜色 (为9个模型分配不同的颜色)
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6'
]


# ================= 功能函数 =================

def load_data_and_calc_roc(model_name, dataset_config):
    """读取特定模型在特定数据集上的数据并计算 ROC"""

    # 构造路径
    # 真值路径: datasets/IRSTD-1k/masks
    gt_dir = os.path.join(GT_ROOT, dataset_config["gt_folder_name"], "masks")
    # 预测路径: predict/GIDNet/IRSTD-1K
    pred_dir = os.path.join(PRED_ROOT, model_name, dataset_config["pred_folder_name"])

    if not os.path.exists(pred_dir):
        print(f"Warning: 路径不存在 {pred_dir}，跳过该模型。")
        return None, None, None

    # 获取文件列表 (确保对应)
    # 假设预测图和掩码图文件名一致 (不含后缀可能不同，这里假设文件名完全一致或可以通过名字匹配)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.bmp', '.jpg'))])

    y_true = []
    y_scores = []

    # 使用 tqdm 显示进度
    desc = f"[{dataset_config['name']}] Loading {model_name}"
    for file_name in tqdm(pred_files, desc=desc, leave=False):
        pred_path = os.path.join(pred_dir, file_name)
        gt_path = os.path.join(gt_dir, file_name)  # 假设文件名完全一样

        # 如果后缀不一样（比如预测是 .png，真值是 .bmp），尝试替换后缀
        if not os.path.exists(gt_path):
            base_name = os.path.splitext(file_name)[0]
            # 尝试寻找对应的 mask 文件
            possible_exts = ['.png', '.bmp', '.jpg', '_mask.png']
            found = False
            for ext in possible_exts:
                temp_path = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(temp_path):
                    gt_path = temp_path
                    found = True
                    break
            if not found:
                continue  # 找不到对应的真值，跳过

        # 读取图片
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img_pred is None or img_gt is None:
            continue

        # 统一大小 (防止尺寸不匹配报错)
        if img_pred.shape != img_gt.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

        # 归一化预测值 (0-1)
        scores = img_pred.astype(np.float32) / 255.0
        # 二值化真值 (0, 1)
        labels = (img_gt > 127).astype(np.int8)

        y_scores.append(scores.flatten())
        y_true.append(labels.flatten())

    if not y_true:
        return None, None, None

    # 拼接所有像素
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)

    # 计算 ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# ================= 主程序 =================

def main():
    # 设置画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 遍历三个数据集
    for i, dataset_cfg in enumerate(DATASETS_CONFIG):
        ax = axes[i]
        dataset_name = dataset_cfg["name"]
        print(f"\n正在处理数据集: {dataset_name} ...")

        # 遍历九个模型
        for j, model_name in enumerate(MODELS):
            fpr, tpr, roc_auc = load_data_and_calc_roc(model_name, dataset_cfg)

            if fpr is not None:
                # 绘制曲线
                ax.plot(fpr, tpr, color=COLORS[j], lw=1.5,
                        label=f'{model_name} ')
            else:
                print(f"  - {model_name}: 无数据")

        # --- 设置当前子图的样式 (参考您的第一张图) ---
        ax.set_title(f'ROC curve on {dataset_name}', fontsize=14)
        ax.set_xlabel('False Positive Ratio (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Ratio (TPR)', fontsize=12)

        # 关键：设置为科学计数法显示 X 轴
        ax.set_xlim([0.0, 1e-4])  # 根据通常 IRSTD 的范围设置，如果曲线挤在一起，改大这个值
        ax.set_ylim([0.0, 1.02])
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    save_path = "comparison_roc_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n绘图完成！图片已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()