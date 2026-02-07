import re


def parse_log(log_content):
    """
    从日志内容中提取验证阶段的指标。
    参数:
        log_content (str): 日志文件内容
    返回:
        list: 包含每个 epoch 的验证指标字典
    """
    # 正则表达式匹配 [Val] 开头的行，提取 epoch, loss, RRE, RTE, RR
    pattern = r'\[Val\] Epoch: (\d+), .* loss: ([\d.]+), .* RRE: ([\d.]+), RTE: ([\d.]+), RR: ([\d.]+)'
    matches = re.findall(pattern, log_content)

    val_data = []
    for match in matches:
        epoch = int(match[0])
        loss = float(match[1])
        rre = float(match[2])
        rte = float(match[3])
        rr = float(match[4])
        val_data.append({
            'epoch': epoch,
            'loss': loss,
            'RRE': rre,
            'RTE': rte,
            'RR': rr
        })
    return val_data


def find_best_model(val_data):
    """
    根据 RR, RRE, RTE 和 loss 找到最佳模型。
    参数:
        val_data (list): 验证指标列表
    返回:
        dict: 最佳模型的指标字典，或 None（如果无数据）
    """
    if not val_data:
        return None

    # 找到最大的 RR 值
    max_rr = max(d['RR'] for d in val_data)
    # 筛选出 RR 达到最大值的候选项
    candidates = [d for d in val_data if d['RR'] == max_rr]

    # 在候选项中，选择 RRE + RTE + loss 最小的模型
    best_model = min(candidates, key=lambda x: x['RRE'] + x['RTE'] + x['loss'])
    return best_model


# 日志文件内容
# with open('/home/fang/Downloads/GeoTransformer-main/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/logs/train-20250316-015630.log', 'r') as f:
#     log_content = f.read()

# with open('/home/fang/Downloads/GeoTransformer-main/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/logs/train-20250315-185129.log', 'r') as f:
#     log_content = f.read()
# with open('/home/fang/Downloads/GeoTransformer-main/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/logs/train-20250315-123102.log', 'r') as f:
#     log_content = f.read()
# with open('/home/fang/Downloads/GeoTransformer-main/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/logs/train-20250315-014636.log', 'r') as f:
#     log_content = f.read()
with open('/home/fang/Downloads/GeoTransformer-main/output/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/logs/train-20250304-002632.log', 'r') as f:
    log_content = f.read()

# 提取验证数据
val_data = parse_log(log_content)

# 寻找最佳模型
best_model = find_best_model(val_data)

# 输出结果
if best_model:
    print(f"最佳模型位于 epoch {best_model['epoch']}:")
    print(f"  RR: {best_model['RR']}")
    print(f"  RRE: {best_model['RRE']}")
    print(f"  RTE: {best_model['RTE']}")
    print(f"  loss: {best_model['loss']}")
else:
    print("未找到验证数据。")
