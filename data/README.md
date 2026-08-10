# 数据放置

仓库保留题目给出的 5 组 256x256 DAPI ROI 与人工二值 Mask，位于
`data/annotations/`。完整图像约 172 MB，不在 Git 中重复分发；运行整图推理时将
下面的文件路径作为参数传入即可：

```text
Y00039K4_DAPI_transformed-c.tif
```

程序按文件名配对 ROI 与标注：

```text
<name>-dapi.tif
<name>-dapi_mask.tif
```

图像应为二维 `uint8` TIFF，Mask 允许 `0/1` 或 `0/255`，读取后统一解释为二值前景。
