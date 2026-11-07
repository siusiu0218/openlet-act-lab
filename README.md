# openlet-act-lab

OpenLET 仿真抓取基线的环境记录与验收工具，以及 ACT 离线评估、检查点发布和验证缓存工具。

## 快速运行

使用 Python 3.10+，在本目录执行。NumPy 是示例与单元测试的唯一第三方依赖。

```sh
python -m pip install -r requirements.txt
python -m examples.offline_metrics
python -m unittest discover -v
```

示例生成合成数据并输出 JSON 指标。示例中的限值仅用于测试。

## 模块与接口

- `offline_metrics.metrics(arrays, records, limits, action_mean)` 输出逐轴 MAE、RMSE、P95、最大误差、位置越界和动作跳变统计，最多35个输入。`report` 增加逐 episode 统计。
- `long_offline_metrics.metrics` 接受 `max_inputs=350`。默认仍为35，调用者须显式指定。
- `inference_contract.py` 提供指标所需的限值验证，并保留原始 CPU 协议辅助函数。ROS集成尚待验收。
- `checkpoint_io.save_checkpoint(destination, latest, total_steps, writers, trace)` 调用写入回调，在暂存目录完成文件SHA与回执后发布检查点，最后更新 latest。返回目标路径。失败会抛异常，可能留下暂存文件供诊断。单写入者使用，既有目标受到保护。`existing_root=True` 逐文件发布，以完成回执判断完整性。未提供断电持久性或任意嵌套目录校验保证。
- `validation_cache.ValidationCache(dataset, indices, directory, torch, cap)` 使用索引访问数据集，在独占目录缓存原始CPU对象，并校验SHA。固定350个索引，单文件10MiB、总量默认3GiB。损坏或超限抛异常。真实使用需提供 PyTorch，缓存目录必须由本作业控制，加载对象使用 `weights_only=False`。测试使用轻量序列化替身。

## 数据约定

`arrays` 包含 float32 的 state `(N,18)`、target/normalized/prediction `(N,100,21)`，以及 bool 的 is_pad `(N,100)`。`records` 每项包含 episode、frame、length。episode 限定24至30。state 顺序为右臂7轴和手部11轴，action 另含底盘3维。

prediction 与 target 应为模型后处理后的数据集单位，手部乘1.57转换为rad，底盘单位为m/s、m/s、rad/s。position和delta限值同样使用数据集单位。normalized 保留作证据字段，当前指标函数不使用它计算归一化损失。MAE等目标误差排除padding，完整动作块越界另外统计。采样、维度及单位都是当前实验的固定契约。

## 验证范围

2026-11-06的来源实验完成4000到10000步训练，固定350输入跨轮评估及指标独立复算通过。缓存直接读取与重复读取输出逐值一致。10000步 normalized L1 为0.164243，首步超过候选位置限值0.01rad的输入为224/350，机器人动作准入失败。350输入已用于模型选择，独立最终测试和ROS闭环仍待验收。

本仓库提供提取模块与合成示例。完整训练启动、模型、数据、集群配置及原始实验产物由来源项目管理。可移植版本验收以本仓库测试为准，历史实验结果属于来源版本。

## 来源与许可

模块从本地 OpenLET 管理仓库提交 `38c50dd` 的 `tools/act-bootstrap` 提取，逐文件SHA见 `docs/source_manifest.json`。它们是围绕 LeRobot ACT 实验编写的适配工具。本次未复制 LeRobot、PyTorch 或机器人官方源码。第三方依赖遵循各自许可证。

仓库所有者尚未选择开源许可证，当前未授予额外再分发许可。

## 训练与恢复示例

配置驱动的ACT入口、恢复契约、350输入评估函数和CPU演示见 [训练说明](docs/training.md)。移植入口为实验性代码，真实ACT运行待目标环境验收。

## 目录结构

```text
openlet_act/   核心模块与训练命令入口
examples/      合成数据与保存恢复演示
configs/       可复制的配置模板
tests/         独立单元测试
docs/          训练说明及来源记录
```

所有示例和测试命令均在仓库根目录运行。个人配置与运行产物由 .gitignore 排除。

## 仿真抓取基线

环境清单位于 `environment/`，动作与退出验收脚本位于 `simulation/`。依赖、运行方式和历史结果见 [仿真说明](docs/simulation.md)。完整仿真部署依赖仍待补齐。
