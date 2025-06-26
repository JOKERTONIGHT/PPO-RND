# PPO-RND Montezuma's Revenge - Updated for Modern Libraries

## 更新内容 (Updates)

本项目已更新以兼容当前主流的机器学习库版本：

### 主要库更新 (Library Updates)

1. **Gym → Gymnasium**: 更新到 Gymnasium >= 0.29.0
   - 处理新的 step() 返回格式 (observation, reward, terminated, truncated, info)
   - 更新 reset() 方法以返回 (observation, info)
   - 添加 ALE-py 支持

2. **PyTorch**: 更新到 >= 2.0.0
   - 移除已废弃的 `torch.ByteTensor`，使用 `torch.tensor(dtype=torch.uint8)`
   - 修复 `torch._six.inf` 引用，使用 `math.inf`

3. **其他依赖更新**:
   - numpy >= 1.24.0
   - opencv-python >= 4.8.0 (移除 opencv-contrib-python)
   - matplotlib >= 3.7.0
   - tensorboard >= 2.13.0

### 代码修改 (Code Changes)

#### 环境相关 (Environment)
- `make_atari()`: 使用 `gym.wrappers.TimeLimit` 替代直接设置 `_max_episode_steps`
- 所有 wrapper 类更新以处理新的 step/reset 格式
- 修复布尔类型使用 (`np.bool` → `bool`)

#### 模型相关 (Model)
- `Brain.choose_mini_batch()`: 更新张量创建方式
- 保持所有模型架构和训练逻辑不变

#### 实用工具 (Utils)
- 更新导入和错误处理
- 修复梯度裁剪函数中的无穷大引用

## 安装指南 (Installation)

```bash
# 安装依赖
pip install -r requirements.txt

# 测试环境
python test_environment.py

# 开始训练
python main.py

# 测试已训练模型
python main.py --do_test
```

## 兼容性说明 (Compatibility)

- 保持原有的项目结构和训练逻辑完全不变
- 所有超参数和算法实现保持一致
- 预训练模型应该仍然兼容（checkpoint 格式未变）

## 主要特性 (Features)

- ✅ PPO (Proximal Policy Optimization) 算法
- ✅ RND (Random Network Distillation) 内在奖励
- ✅ 多进程并行环境
- ✅ Tensorboard 日志记录
- ✅ 模型检查点保存/加载
- ✅ 现代库兼容性

## 故障排除 (Troubleshooting)

如果遇到问题，请检查：

1. **ALE 游戏 ROM**: 确保安装了 Montezuma's Revenge ROM
2. **CUDA 支持**: 如果有 NVIDIA GPU，确保安装了兼容的 PyTorch CUDA 版本
3. **依赖版本**: 运行 `pip list` 检查所有依赖版本是否正确

更多详细信息请参考原始 README.md 文件。
