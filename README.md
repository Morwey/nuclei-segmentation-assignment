# DAPI 细胞核分割

本仓库完成一张 15000x11496 DAPI 全图的细胞核二值分割，并在 5 个带人工 Mask 的
256x256 ROI 上比较 Cellpose-SAM 与 nnU-Net v2。两条路线分别完成交叉验证、全量
训练和整图推理。

完整实验设计和结果见 [技术报告](docs/REPORT.md)；排版版见
[PDF](docs/REPORT.pdf)。

## ROI 结果

表中结果为 5 个 ROI 的宏平均；微调和从头训练采用 5 折留一 ROI。FNR 和 FDR
均按像素计算。

| 方法 | Dice | FNR | FDR | FNR、FDR 均低于 10% 的 ROI |
|---|---:|---:|---:|---:|
| Cellpose-SAM 直接推理 | 0.9152 | 5.70% | 10.76% | 1/5 |
| Cellpose-SAM 微调 | **0.9259** | **6.61%** | **7.95%** | 2/5 |
| nnU-Net v2 从头训练 | 0.9130 | 8.38% | 8.73% | 2/5 |

![指标比较](results/figures/metric_comparison.png)

## 整图结果

Cellpose-SAM 和 nnU-Net 分别使用全部 5 个 ROI 训练，并输出同尺寸二值 TIFF：

| 方法 | 完整 Mask | 文件信息 | 抽检图 |
|---|---|---|---|
| Cellpose-SAM | [Mask](results/full_image/cellpose_sam_mask.tif) | [JSON](results/full_image/cellpose_sam_mask.json) | [PNG](results/figures/cellpose_sam_full_overview.png) |
| nnU-Net v2 | [Mask](results/full_image/nnunet_mask.tif) | [JSON](results/full_image/nnunet_mask.json) | [PNG](results/figures/nnunet_full_overview.png) |

两种方法在相同位置的局部结果见
[整图抽检对比](results/figures/full_image_method_comparison.png)。

## 目录

```text
.
├── configs/                 # 模型参数和交叉验证划分
├── data/annotations/        # 5 组 ROI 与人工二值标注
├── docs/                    # Markdown 与 PDF 实验报告
├── results/
│   ├── full_image/          # 两种方法的完整 Mask 与元数据
│   ├── roi_predictions/     # 5 折留出预测
│   ├── figures/             # 指标、ROI 和整图抽检图
│   └── metrics_*.csv/json   # 逐图与汇总指标
├── scripts/                 # 训练、评估和整图推理入口
├── src/                     # 数据处理、模型流程和指标代码
└── tests/                   # 指标单元测试
```

原始全图和模型权重不放入仓库。全图从作业数据目录读取；Cellpose-SAM 权重由
`train-all` 生成，nnU-Net 权重由全量训练生成。

## 环境与复现

建议使用 Python 3.12。复核已有 Mask 和指标：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/evaluate.sh
```

Cellpose-SAM：

```bash
pip install -r requirements-cellpose.txt
./scripts/run_cellpose.sh zero-shot
./scripts/run_cellpose.sh cross-validation
./scripts/run_cellpose.sh train-all
./scripts/infer_cellpose_full.sh /absolute/path/to/Y00039K4_DAPI_transformed-c.tif
```

nnU-Net v2：

```bash
pip install -r requirements-nnunet.txt
./scripts/run_nnunet.sh
./scripts/infer_nnunet_full.sh /absolute/path/to/Y00039K4_DAPI_transformed-c.tif
```

同时运行两条整图路线：

```bash
./scripts/infer_full_image.sh /absolute/path/to/Y00039K4_DAPI_transformed-c.tif
```

设备默认自动选择。可通过 `CELLPOSE_DEVICE`、`CELLPOSE_BATCH_SIZE` 和
`NNUNET_DEVICE` 调整推理设备与批量大小。
