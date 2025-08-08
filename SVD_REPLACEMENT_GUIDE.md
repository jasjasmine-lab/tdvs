# iSVD层替换指南

本文档详细说明了项目中所有使用iSVD（增量奇异值分解）的函数位置，以及如何将其替换为其他SVD方法。

## 需要修改的文件和函数

### 1. `cdm/gpm.py` - 主要的iSVD实现文件

#### 函数1: `on_test_batch_end()`
- **位置**: 第69-102行
- **原功能**: 每10个批次进行一次SVD分解，保留99.9%的能量
- **SVD调用**: `torch.linalg.svd(act.cuda(), full_matrices=False)`
- **修改点**: 可替换为randomized SVD或TruncatedSVD

#### 函数2: `on_test_end()` ⭐ **核心函数**
- **位置**: 第104-171行
- **原功能**: 实现增量SVD的核心逻辑
- **SVD调用**: 
  - `torch.linalg.svd(act.cuda(), full_matrices=False)` (第一次)
  - `torch.linalg.svd(act_hat)` (第二次，对正交化后的激活)
- **修改点**: 这是iSVD的核心，可替换为IncrementalPCA或重新计算所有投影

### 2. `cdm/sd_amn.py` - 被注释的iSVD代码

#### 函数3: `on_test_batch_end()` (已注释)
- **位置**: 第219-233行
- **原功能**: 与gpm.py中的功能相同
- **状态**: 当前被注释，如果启用需要同样修改

#### 函数4: `on_test_end()` (已注释)
- **位置**: 第234-280行
- **原功能**: 与gpm.py中的功能相同
- **状态**: 当前被注释，如果启用需要同样修改

## SVD替换方案

### 方案1: 使用PyTorch的随机化SVD
```python
# 替换 torch.linalg.svd(act.cuda(), full_matrices=False)
# 为:
U, S, Vh = torch.svd_lowrank(act.cuda(), q=min(act.shape)-1)
```

### 方案2: 使用sklearn的TruncatedSVD
```python
from sklearn.decomposition import TruncatedSVD
n_components = min(min(act.shape)-1, 100)
svd = TruncatedSVD(n_components=n_components)
U_truncated = svd.fit_transform(act.cuda().cpu().numpy())
self.act[name] = torch.tensor(U_truncated).float()
```

### 方案3: 使用IncrementalPCA (推荐用于增量学习)
```python
from sklearn.decomposition import IncrementalPCA
if not hasattr(self, 'ipca_models'):
    self.ipca_models = {}
if name not in self.ipca_models:
    self.ipca_models[name] = IncrementalPCA(n_components=min(act.shape[0], 100))
    self.project[name] = torch.tensor(self.ipca_models[name].fit_transform(act.cuda().cpu().numpy().T)).cuda().T
else:
    self.ipca_models[name].partial_fit(act.cuda().cpu().numpy().T)
    self.project[name] = torch.tensor(self.ipca_models[name].components_).cuda()
```

### 方案4: 简单的标准SVD重新计算
```python
# 将新激活与已有投影合并后重新计算SVD
combined_act = torch.cat([self.project[name], act.cuda()], dim=1)
U_new, S_new, Vh_new = torch.linalg.svd(combined_act, full_matrices=False)
# 根据能量阈值选择维度
sval_total_new = (S_new**2).sum()
sval_ratio_new = (S_new**2) / sval_total_new
r_new = max(torch.sum(torch.cumsum(sval_ratio_new, dim=0) < 0.99), 1)
self.project[name] = U_new[:, :r_new]
```

## 需要添加的依赖

如果选择使用sklearn方法，需要在文件顶部添加：
```python
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.utils.extmath import randomized_svd
import numpy as np
```

## 修改建议

1. **优先修改 `cdm/gpm.py`**: 这是当前活跃使用的文件
2. **测试性能**: 不同SVD方法的计算复杂度和精度不同
3. **保持接口一致**: 确保替换后的方法返回相同格式的数据
4. **考虑内存使用**: 某些方法可能需要更多内存
5. **备份原始代码**: 在修改前备份原始实现

## 性能对比

| 方法 | 计算速度 | 内存使用 | 精度 | 增量学习 |
|------|----------|----------|------|----------|
| torch.linalg.svd | 慢 | 高 | 高 | 否 |
| torch.svd_lowrank | 快 | 中 | 中 | 否 |
| TruncatedSVD | 快 | 低 | 中 | 否 |
| IncrementalPCA | 中 | 低 | 中 | 是 |

## 注意事项

- iSVD的核心优势是增量学习能力，替换时需要考虑是否保持这一特性
- 能量阈值（0.999, 0.5等）可能需要根据新方法调整
- 某些方法可能需要转换数据格式（torch tensor ↔ numpy array）
- 测试时建议先在小数据集上验证替换效果